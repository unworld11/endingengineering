from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import utils.utils as util
import utils.quantization as q

import numpy as np
import os, time, sys
import copy
import argparse

#########################
# supported model candidates
candidates = [
    'binput-pg',
    'adaptive-pg',      # Adaptive PG (Ada-FracBNN)
    'adaptive-pg-kd',   # Adaptive PG with Knowledge Distillation
]
#########################


#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--model_id', '-id', type=int, default=0)
parser.add_argument('--gtarget', '-g', type=float, default=0.0)
parser.add_argument('--init_lr', '-lr', type=float, default=1e-3)
parser.add_argument('--batch_size', '-b', type=int, default=128)
parser.add_argument('--num_epoch', '-e', type=int, default=250)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5)
parser.add_argument('--last_epoch', '-last', type=int, default=-1)
parser.add_argument('--finetune', '-f', action='store_true', help='finetune the model')
parser.add_argument('--save', '-s', action='store_true', help='save the model')
parser.add_argument('--test', '-t', action='store_true', help='test only')
parser.add_argument('--resume', '-r', type=str, default=None,
                    help='path of the model checkpoint for resuming training')
parser.add_argument('--data_dir', '-d', type=str, default='/tmp/cifar10_data',
                    help='path to the dataset directory')
parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')
parser.add_argument('--target_sparsity', '-ts', type=float, default=0.15,
                    help='target sparsity for adaptive PG (0.1-0.2)')
parser.add_argument('--sparsity_weight', '-sw', type=float, default=0.01,
                    help='weight for sparsity regularization loss')
parser.add_argument('--kd_temperature', '-temp', type=float, default=4.0,
                    help='temperature for knowledge distillation')
parser.add_argument('--kd_alpha', '-alpha', type=float, default=0.7,
                    help='alpha for knowledge distillation loss weighting')
parser.add_argument('--teacher_path', '-tp', type=str, default=None,
                    help='path to pretrained teacher model')
parser.add_argument('--entropy_weight', '-ew', type=float, default=0.0,
                    help='weight for gate entropy regularization (encourage 0/1 gates)')
parser.add_argument('--pg_kd', action='store_true',
                    help='enable PG-only feature KD on upgraded channels')
parser.add_argument('--pg_kd_weight', type=float, default=0.5,
                    help='weight for PG-only feature KD loss (feature MSE)')

args = parser.parse_args()

# Validate model_id
if args.model_id < 0 or args.model_id >= len(candidates):
    raise ValueError(f"Invalid model_id={args.model_id}. Must be in range [0, {len(candidates)-1}]. "
                     f"Available models: {', '.join([f'{i}={c}' for i, c in enumerate(candidates)])}")

_ARCH = candidates[args.model_id]
# All our models use binary input encoder, so need drop_last=True
drop_last = True if ('binput' in _ARCH or 'adaptive' in _ARCH) else False


#----------------------------
# Load the CIFAR-10 dataset.
#----------------------------
def load_cifar10():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ]
    transform_test_list = [transforms.ToTensor()]

    # Don't normalize for models with binary input encoder (binput and adaptive-pg)
    if 'binput' not in _ARCH and 'adaptive' not in _ARCH:
        transform_train_list.append(normalize)
        transform_test_list.append(normalize)
        print("Using normalized inputs (ImageNet stats) for data preprocessing")
    else:
        print("Using raw [0,1] inputs for binary input encoder (no normalization)")

    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)

    # pin_memory=True makes transferring data from host to GPU faster
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2,
                                              pin_memory=True, drop_last=drop_last)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2,
                                             pin_memory=True, drop_last=drop_last)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


#----------------------------
# Define the model.
#----------------------------
def generate_model(model_arch):
    import model.fracbnn_cifar10 as m
    
    if 'binput-pg' == model_arch:
        return m.resnet20(batch_size=args.batch_size, num_gpus=torch.cuda.device_count())
    elif 'adaptive-pg' in model_arch:
        return m.resnet20(batch_size=args.batch_size, num_gpus=torch.cuda.device_count(),
                         adaptive_pg=True, target_sparsity=args.target_sparsity)
    else:
        raise NotImplementedError("Model architecture is not supported.")

def generate_teacher_model():
    """Generate teacher model for knowledge distillation"""
    import model.fracbnn_cifar10 as m
    return m.fp_resnet20(num_classes=10)


#----------------------------
# Train the network.
#----------------------------
def train_model(trainloader, testloader, net,
                optimizer, scheduler, start_epoch, device, teacher_net=None):
    # define the loss function
    criterion = (nn.CrossEntropyLoss().cuda()
                 if torch.cuda.is_available() else nn.CrossEntropyLoss())
    
    # Knowledge distillation setup
    use_kd = teacher_net is not None and 'kd' in _ARCH
    normalize_teacher_inputs = False
    if use_kd:
        import utils.quantization as q
        kd_criterion = q.KnowledgeDistillationLoss(
            temperature=args.kd_temperature, 
            alpha=args.kd_alpha
        ).to(device)
        teacher_net.eval()  # Teacher always in eval mode
        print(f"Using Knowledge Distillation with T={args.kd_temperature}, alpha={args.kd_alpha}")
        
        # Pre-compute normalization tensors for teacher (if student uses binary input)
        if 'adaptive' in _ARCH or 'binput' in _ARCH:
            normalize_teacher_inputs = True
            normalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            normalize_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            print("Teacher will receive normalized inputs, student receives raw [0,1] inputs")

    # PG-only KD feature hooks (student and teacher) when enabled
    use_pg_kd = bool(args.pg_kd)
    student_feats = {}
    teacher_feats = {}
    student_handles = []
    teacher_handles = []

    def register_pg_feature_hooks(model, is_teacher=False):
        feats = teacher_feats if is_teacher else student_feats
        handles = teacher_handles if is_teacher else student_handles

        def make_hook(name):
            def hook(module, inp, out):
                feats[name] = out.detach() if is_teacher else out
            return hook

        for name, module in model.named_modules():
            if isinstance(module, q.PGBinaryConv2d) and getattr(module, 'adaptive_pg', False):
                handles.append(module.register_forward_hook(make_hook(name)))

    if use_pg_kd and teacher_net is not None:
        import utils.quantization as q
        register_pg_feature_hooks(net, is_teacher=False)
        register_pg_feature_hooks(teacher_net, is_teacher=True)
        print("Enabled PG-only feature KD with hooks on adaptive PG layers.")

    best_acc = 0.0
    best_model = copy.deepcopy(net.state_dict())

    for epoch in range(start_epoch, args.num_epoch):  # loop over the dataset multiple times

        # set printing functions
        batch_time = util.AverageMeter('Time/batch', ':.2f')
        losses = util.AverageMeter('Loss', ':6.2f')
        top1 = util.AverageMeter('Acc', ':6.2f')
        progress = util.ProgressMeter(
            len(trainloader),
            [losses, top1, batch_time],
            prefix="Epoch: [{}]".format(epoch + 1)
        )

        # switch the model to the training mode
        net.train()

        print('current learning rate = {}'.format(optimizer.param_groups[0]['lr']))

        # each epoch
        end = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            
            # Base loss computation
            if use_kd:
                # Knowledge distillation loss
                with torch.no_grad():
                    if normalize_teacher_inputs:
                        # Teacher needs normalized inputs, student uses raw [0,1] inputs
                        teacher_inputs = (inputs - normalize_mean) / normalize_std
                    else:
                        teacher_inputs = inputs
                    teacher_outputs = teacher_net(teacher_inputs)
                loss = kd_criterion(outputs, teacher_outputs, labels)
            else:
                # Standard cross-entropy loss
                loss = criterion(outputs, labels)
            
            # Add regularization losses
            if 'pg' in _ARCH:
                # Original PG regularization
                for name, param in net.named_parameters():
                    if 'threshold' in name:
                        loss += (0.00001 * 0.5 *
                                 torch.norm(param - args.gtarget) *
                                 torch.norm(param - args.gtarget))
                
                # Adaptive PG sparsity regularization
                if 'adaptive-pg' in _ARCH and hasattr(net, 'get_sparsity_loss'):
                    sparsity_loss = net.get_sparsity_loss()
                    loss += args.sparsity_weight * sparsity_loss

                # Gate entropy regularizer (encourage crisper 0/1 gates)
                if 'adaptive-pg' in _ARCH and hasattr(net, 'get_entropy_loss') and args.entropy_weight > 0.0:
                    entropy_loss = net.get_entropy_loss()
                    loss += args.entropy_weight * entropy_loss

                # PG-only feature KD (feature MSE on upgraded channels only)
                if use_pg_kd and teacher_net is not None and student_feats and teacher_feats:
                    pg_kd_loss = 0.0
                    count = 0
                    for name, s_feat in student_feats.items():
                        if name in teacher_feats:
                            t_feat = teacher_feats[name]
                            # Resize teacher feats if DataParallel wraps
                            if t_feat.size() != s_feat.size():
                                # Best-effort: center crop or interpolate spatial dims
                                _, _, h, w = s_feat.size()
                                t_feat = torch.nn.functional.interpolate(t_feat, size=(h, w), mode='bilinear', align_corners=False)
                            # Build channel mask from current module's hard gates
                            module = dict(net.named_modules())[name]
                            if hasattr(module, 'get_hard_gates'):
                                hard = module.get_hard_gates()
                                if hard is not None:
                                    mask = hard.view(1, -1, 1, 1).to(s_feat.device)
                                    # MSE only on upgraded channels
                                    diff = (s_feat - t_feat) * mask
                                    pg_kd_loss = pg_kd_loss + torch.mean(diff * diff)
                                    count += 1
                    if count > 0:
                        loss += args.pg_kd_weight * (pg_kd_loss / count)
            
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            _, batch_predicted = torch.max(outputs.data, 1)
            batch_accu = 100.0 * (batch_predicted == labels).sum().item() / labels.size(0)
            losses.update(loss.item(), labels.size(0))
            top1.update(batch_accu, labels.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 99:
                # print statistics every 100 mini-batches each epoch
                progress.display(i)  # i = batch id in the epoch

        # update the learning rate
        scheduler.step()

        # Temperature annealing for adaptive PG
        if 'adaptive-pg' in _ARCH and hasattr(net, 'set_temperature'):
            # Anneal temperature from 5.0 to 1.0 over training
            temp = 5.0 - 4.0 * (epoch / args.num_epoch)
            temp = max(temp, 1.0)
            net.set_temperature(temp)
        
        # print test accuracy every few epochs
        if epoch % 1 == 0:
            print('epoch {}'.format(epoch + 1))
            epoch_acc = test_accu(testloader, net, device)
            if 'pg' in _ARCH:
                sparsity(testloader, net, device)
            
            # Report adaptive PG gate statistics
            if 'adaptive-pg' in _ARCH and hasattr(net, 'get_gate_statistics'):
                gate_stats = net.get_gate_statistics()
                if gate_stats:
                    avg_active = np.mean([s['active_fraction'] for s in gate_stats])
                    print(f'Average 2-bit fraction: {avg_active:.3f} (target: {args.target_sparsity:.3f})')
            
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(net.state_dict())
            print("The best test accuracy so far: {:.1f}".format(best_acc))

            # save the model if required
            if args.save:
                print("Saving the trained model and states.")
                this_file_path = os.path.dirname(os.path.abspath(__file__))
                save_folder = os.path.join(this_file_path, 'save_CIFAR10_model')
                util.save_models(best_model, save_folder,
                                 suffix=_ARCH + '-finetune' if args.finetune else _ARCH)
                """
                states = {'epoch':epoch+1, 
                          'optimizer':optimizer.state_dict(), 
                          'scheduler':scheduler.state_dict()}
                util.save_states(states, save_folder, suffix=_ARCH)
                """

    print('Finished Training')


#----------------------------
# Test accuracy.
#----------------------------
def test_accu(testloader, net, device):
    correct = 0
    total = 0
    # switch the model to the evaluation mode
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print('Accuracy of the network on the 10000 test images: %.1f %%' % accuracy)
    return accuracy


#----------------------------
# Report sparsity in PG  (robust to 1-D / 2-D shapes)
#----------------------------
def sparsity(testloader, net, device):
    """
    Collects per-layer PG counters (num_out, num_high) after each forward pass
    and accumulates them as scalar totals. This avoids NumPy broadcasting issues
    when layers report lists, 1-D arrays, or 2-D arrays.
    """
    def collect_pg_counters():
        outs, highs = [], []
        def grab(m):
            if isinstance(m, q.PGBinaryConv2d):
                # Convert whatever we get to numpy arrays, then sum later.
                outs.append(np.asarray(getattr(m, 'num_out', 0)))
                highs.append(np.asarray(getattr(m, 'num_high', 0)))
        net.apply(grab)
        return outs, highs

    net.eval()
    total_out = 0.0
    total_high = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            _ = net(images)  # forward to populate counters
            outs, highs = collect_pg_counters()
            if outs:
                total_out += float(np.sum(outs))
            if highs:
                total_high += float(np.sum(highs))

    sparsity_pct = 100.0 - (total_high / total_out) * 100.0 if total_out > 0 else 0.0
    print('Sparsity of the update phase: %.1f %%' % sparsity_pct)


#----------------------------
# Remove the saved placeholder
#----------------------------
def remove_placeholder(state_dict):
    from collections import OrderedDict
    temp_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if 'encoder.placeholder' in key:
            pass
        else:
            temp_state_dict[key] = value
    return temp_state_dict


#----------------------------
# Metrics and Analysis
#----------------------------
def analyze_adaptive_pg_metrics(net, testloader, device):
    """Analyze and report metrics for Adaptive PG model"""
    if not ('adaptive-pg' in _ARCH and hasattr(net, 'get_gate_statistics')):
        return
    
    print("\n" + "="*50)
    print("ADAPTIVE PG (Ada-FracBNN) ANALYSIS")
    print("="*50)
    
    # Gate statistics
    gate_stats = net.get_gate_statistics()
    if gate_stats:
        print("\nGate Statistics per Layer:")
        total_active = 0
        total_channels = 0
        
        for stats in gate_stats:
            layer_name = stats['layer_name']
            active_frac = stats['active_fraction']
            gate_mean = stats['gate_mean']
            gate_std = stats['gate_std']
            
            print(f"  {layer_name}: 2-bit fraction={active_frac:.3f}, "
                  f"gate_mean={gate_mean:.3f}, gate_std={gate_std:.3f}")
            
            # Estimate number of channels (rough approximation)
            if 'conv1' in layer_name:
                channels = 16 if 'layer1' in layer_name else (32 if 'layer2' in layer_name else 64)
            elif 'conv2' in layer_name:
                channels = 16 if 'layer1' in layer_name else (32 if 'layer2' in layer_name else 64)
            else:
                channels = 16  # fallback
                
            total_active += active_frac * channels
            total_channels += channels
        
        overall_frac = total_active / total_channels if total_channels > 0 else 0
        print(f"\nOverall 2-bit fraction: {overall_frac:.3f}")
        print(f"Target sparsity: {args.target_sparsity:.3f}")
        print(f"Sparsity achieved: {1.0 - overall_frac:.3f}")
    
    # Compute rough FLOP estimates
    print(f"\nCompute Analysis (rough estimates):")
    print(f"  Standard BNN (1-bit): ~100% 1-bit operations")
    print(f"  Ada-FracBNN: ~{(1.0-overall_frac)*100:.1f}% 1-bit, ~{overall_frac*100:.1f}% 2-bit operations")
    print(f"  Compute overhead: ~{overall_frac*100:.1f}% increase in effective operations")


#----------------------------
# Main function.
#----------------------------
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Available GPUs: {}".format(torch.cuda.device_count()))

    print("Create {} model.".format(_ARCH))
    net = generate_model(_ARCH)

    if torch.cuda.device_count() > 1:
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        print("Activate multi GPU support.")
        net = nn.DataParallel(net)
    net.to(device)
    
    # Initialize teacher model for knowledge distillation
    teacher_net = None
    if 'kd' in _ARCH:
        print("Loading teacher model for knowledge distillation...")
        teacher_net = generate_teacher_model()
        if args.teacher_path and os.path.exists(args.teacher_path):
            print(f"Loading pretrained teacher from {args.teacher_path}")
            teacher_state = torch.load(args.teacher_path)
            teacher_net.load_state_dict(teacher_state, strict=False)
            print("✓ Successfully loaded pretrained teacher weights")
        else:
            print("="*60)
            print("⚠️  WARNING: No pretrained teacher provided!")
            print("   Using randomly initialized teacher for KD.")
            print("   For best results, pretrain a teacher model first or use --teacher_path")
            print("="*60)
        
        if torch.cuda.device_count() > 1:
            teacher_net = nn.DataParallel(teacher_net)
        teacher_net.to(device)

    #------------------
    # Load model params
    #------------------
    if args.resume is not None:
        model_path = args.resume
        if os.path.exists(model_path):
            print("@ Load trained model from {}.".format(model_path))
            state_dict = torch.load(model_path)
            state_dict = remove_placeholder(state_dict)
            net.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Model not found.")

    #-----------------
    # Prepare Data
    #-----------------
    print("Loading the data.")
    trainloader, testloader, classes = load_cifar10()

    #-----------------
    # Test
    #-----------------
    if args.test:
        print("Mode: Test only.")
        test_accu(testloader, net, device)
        if 'pg' in _ARCH:
            sparsity(testloader, net, device)

    #-----------------
    # Finetune
    #-----------------
    elif args.finetune:
        print("num epochs = {}".format(args.num_epoch))
        initial_lr = args.init_lr
        print("init lr = {}".format(initial_lr))
        optimizer = optim.Adam(net.parameters(),
                               lr=initial_lr,
                               weight_decay=0.)
        lr_decay_milestones = [100, 150, 200]
        print("milestones = {}".format(lr_decay_milestones))
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_decay_milestones,
            gamma=0.1,
            last_epoch=args.last_epoch)
        start_epoch = 0
        print("Start finetuning.")
        train_model(trainloader, testloader, net,
                    optimizer, scheduler, start_epoch, device, teacher_net)
        test_accu(testloader, net, device)

    #-----------------
    # Train
    #-----------------
    else:
        print("num epochs = {}".format(args.num_epoch))
        #-----------
        # Optimizer
        #-----------
        initial_lr = args.init_lr
        optimizer = optim.Adam(net.parameters(),
                               lr=initial_lr,
                               weight_decay=args.weight_decay)

        #-----------
        # Scheduler
        #-----------
        print("Use linear learning rate decay.")
        lambda1 = lambda epoch: (1.0 - epoch / args.num_epoch)  # linear decay
        # lambda1 = lambda epoch : (0.7**epoch) # exponential decay
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda1,
            last_epoch=args.last_epoch)

        start_epoch = 0
        print("Start training.")
        train_model(trainloader, testloader, net,
                    optimizer, scheduler, start_epoch, device, teacher_net)
        final_acc = test_accu(testloader, net, device)
        
        # Analyze adaptive PG metrics
        analyze_adaptive_pg_metrics(net, testloader, device)


if __name__ == "__main__":
    main()
