from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
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
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10/100 Training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100'],
                    help='dataset to train on (default: cifar10)')
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
parser.add_argument('--pg_kd_weight', type=float, default=0.1,
                    help='weight for PG-only feature KD loss (feature MSE)')
parser.add_argument('--label_smoothing', type=float, default=0.05,
                    help='label smoothing for hard-label supervision')
parser.add_argument('--mixup_alpha', type=float, default=None,
                    help='mixup alpha; default is 0.2 for adaptive models and 0.0 otherwise')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='EMA decay for validation/save model tracking; set <=0 to disable')
parser.add_argument('--scheduler', type=str, default='cosine',
                    choices=['linear', 'cosine'],
                    help='learning rate scheduler for training')
parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='warmup epochs used by cosine scheduler')
parser.add_argument('--num_workers', type=int, default=None,
                    help='dataloader workers; default chooses a sensible value for the host')
parser.add_argument('--eval_interval', type=int, default=5,
                    help='run validation every N epochs (final epoch always runs)')
parser.add_argument('--sparsity_interval', type=int, default=10,
                    help='report sparsity every N epochs when validating (final epoch always runs)')

args = parser.parse_args()

# Validate model_id
if args.model_id < 0 or args.model_id >= len(candidates):
    raise ValueError(f"Invalid model_id={args.model_id}. Must be in range [0, {len(candidates)-1}]. "
                     f"Available models: {', '.join([f'{i}={c}' for i, c in enumerate(candidates)])}")

_ARCH = candidates[args.model_id]
NUM_CLASSES = 100 if args.dataset == 'cifar100' else 10
# InputEncoder now supports dynamic batch sizes, so keep all samples.
drop_last = False


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def resolve_num_workers():
    if args.num_workers is not None:
        return max(0, args.num_workers)
    cpu_count = os.cpu_count() or 2
    return max(2, min(8, cpu_count))


def resolve_mixup_alpha():
    if args.mixup_alpha is not None:
        return max(0.0, args.mixup_alpha)
    return 0.2 if 'adaptive-pg' in _ARCH else 0.0


def build_scheduler(optimizer):
    if args.scheduler == 'linear':
        print("Use linear learning rate decay.")
        lambda1 = lambda epoch: (1.0 - epoch / args.num_epoch)
    else:
        print(f"Use cosine learning rate decay with {args.warmup_epochs} warmup epochs.")

        def lambda1(epoch):
            current_epoch = epoch + 1
            if args.warmup_epochs > 0 and current_epoch <= args.warmup_epochs:
                return current_epoch / float(max(1, args.warmup_epochs))

            progress = (current_epoch - args.warmup_epochs) / float(max(1, args.num_epoch - args.warmup_epochs))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda1,
        last_epoch=args.last_epoch,
    )


def apply_mixup(inputs, labels, num_classes=NUM_CLASSES):
    alpha = resolve_mixup_alpha()
    if alpha <= 0.0:
        return inputs, labels, None

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    soft_targets = F.one_hot(labels, num_classes=num_classes).float()
    mixed_targets = lam * soft_targets + (1.0 - lam) * soft_targets[index]
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    return mixed_inputs, labels, mixed_targets


def should_run_interval(epoch_idx, interval):
    epoch_num = epoch_idx + 1
    if epoch_num == args.num_epoch:
        return True
    if interval <= 0:
        return False
    return (epoch_num % interval) == 0


def should_run_eval(epoch_idx):
    if should_run_interval(epoch_idx, args.eval_interval):
        return True
    tail_epochs = min(20, args.num_epoch)
    return (epoch_idx + 1) > (args.num_epoch - tail_epochs)


#----------------------------
# Load the CIFAR dataset.
#----------------------------
def load_dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ]
    transform_test_list = [transforms.ToTensor()]

    if 'binput' not in _ARCH and 'adaptive' not in _ARCH:
        transform_train_list.append(normalize)
        transform_test_list.append(normalize)
        print("Using normalized inputs (ImageNet stats) for data preprocessing")
    else:
        print("Using raw [0,1] inputs for binary input encoder (no normalization)")

    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)
    num_workers = resolve_num_workers()
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'drop_last': drop_last,
        'persistent_workers': num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 4

    dataset_cls = torchvision.datasets.CIFAR100 if args.dataset == 'cifar100' else torchvision.datasets.CIFAR10
    print(f"Loading {args.dataset.upper()} ({NUM_CLASSES} classes)")

    trainset = dataset_cls(root=args.data_dir, train=True,
                           download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, **loader_kwargs)

    testset = dataset_cls(root=args.data_dir, train=False,
                          download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False, **loader_kwargs)

    print(f"Using {num_workers} dataloader workers")

    return trainloader, testloader


#----------------------------
# Define the model.
#----------------------------
def generate_model(model_arch):
    import model.fracbnn_cifar10 as m
    num_gpus = max(1, torch.cuda.device_count())
    
    if 'binput-pg' == model_arch:
        return m.resnet20(num_classes=NUM_CLASSES, batch_size=args.batch_size, num_gpus=num_gpus)
    elif 'adaptive-pg' in model_arch:
        return m.resnet20(num_classes=NUM_CLASSES, batch_size=args.batch_size, num_gpus=num_gpus,
                         adaptive_pg=True, target_sparsity=args.target_sparsity)
    else:
        raise NotImplementedError("Model architecture is not supported.")

def generate_teacher_model():
    """Generate teacher model for knowledge distillation"""
    import model.fracbnn_cifar10 as m
    return m.fp_resnet20(num_classes=NUM_CLASSES)


#----------------------------
# Train the network.
#----------------------------
def train_model(trainloader, testloader, net,
                optimizer, scheduler, start_epoch, device, teacher_net=None):
    # define the loss function
    criterion = q.LabelSmoothingCrossEntropy(
        label_smoothing=args.label_smoothing
    ).to(device)
    
    # Knowledge distillation setup
    use_kd = teacher_net is not None and 'kd' in _ARCH
    normalize_teacher_inputs = False
    if use_kd:
        kd_criterion = q.KnowledgeDistillationLoss(
            temperature=args.kd_temperature, 
            alpha=args.kd_alpha,
            label_smoothing=args.label_smoothing,
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
    use_pg_kd = bool(args.pg_kd) and use_kd
    student_feats = {}
    teacher_feats = {}
    student_handles = []
    teacher_handles = []
    student_pg_blocks = []
    teacher_pg_blocks = []

    def register_pg_feature_hooks(student_model, teacher_model):
        student_blocks = [
            module for _, module in student_model.named_modules()
            if module.__class__.__name__ == 'BasicBlock'
        ]
        teacher_blocks = [
            module for _, module in teacher_model.named_modules()
            if module.__class__.__name__ == 'FPBasicBlock'
        ]
        if len(student_blocks) != len(teacher_blocks) or len(student_blocks) == 0:
            print("Skipping PG-only feature KD: incompatible student/teacher block layouts.")
            return [], []

        def make_hook(store, key, detach=False):
            def hook(module, inp, out):
                store[key] = out.detach() if detach else out
            return hook

        for idx, (student_block, teacher_block) in enumerate(zip(student_blocks, teacher_blocks)):
            key = f'block_{idx}'
            student_handles.append(student_block.register_forward_hook(
                make_hook(student_feats, key, detach=False)
            ))
            teacher_handles.append(teacher_block.register_forward_hook(
                make_hook(teacher_feats, key, detach=True)
            ))
        return student_blocks, teacher_blocks

    if use_pg_kd:
        student_pg_blocks, teacher_pg_blocks = register_pg_feature_hooks(
            unwrap_model(net), unwrap_model(teacher_net)
        )
        if student_pg_blocks:
            print("Enabled PG-only feature KD with residual block feature alignment.")

    ema_model = None
    if args.ema_decay > 0.0:
        ema_model = util.ModelEma(unwrap_model(net), decay=args.ema_decay)
        print(f"Using EMA for evaluation with decay={args.ema_decay}")

    best_acc = 0.0
    best_model = copy.deepcopy(unwrap_model(net).state_dict())
    best_epoch = 0

    for epoch in range(start_epoch, args.num_epoch):  # loop over the dataset multiple times
        model_ref = unwrap_model(net)
        if 'adaptive-pg' in _ARCH and hasattr(model_ref, 'set_temperature'):
            # Anneal temperature from 5.0 to 1.0 over training, starting at epoch 1.
            progress = epoch / max(1, args.num_epoch - 1)
            temp = 5.0 - 4.0 * progress
            model_ref.set_temperature(max(temp, 1.0))

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
            student_feats.clear()
            teacher_feats.clear()
            inputs = data[0].to(device, non_blocking=True)
            labels = data[1].to(device, non_blocking=True)
            inputs, hard_labels, soft_targets = apply_mixup(inputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

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
                target_for_loss = soft_targets if soft_targets is not None else hard_labels
                loss = kd_criterion(outputs, teacher_outputs, target_for_loss)
            else:
                # Standard cross-entropy loss
                target_for_loss = soft_targets if soft_targets is not None else hard_labels
                loss = criterion(outputs, target_for_loss)
            
            # Add regularization losses
            if 'pg' in _ARCH:
                model_ref = unwrap_model(net)
                if 'adaptive-pg' not in _ARCH:
                    for name, param in model_ref.named_parameters():
                        if 'threshold' in name:
                            loss += (0.00001 * 0.5 *
                                     torch.norm(param - args.gtarget) *
                                     torch.norm(param - args.gtarget))
                
                # Adaptive PG sparsity regularization
                if 'adaptive-pg' in _ARCH and hasattr(model_ref, 'get_sparsity_loss'):
                    sparsity_loss = model_ref.get_sparsity_loss()
                    loss += args.sparsity_weight * sparsity_loss

                # Gate entropy regularizer (encourage crisper 0/1 gates)
                if 'adaptive-pg' in _ARCH and hasattr(model_ref, 'get_entropy_loss') and args.entropy_weight > 0.0:
                    entropy_loss = model_ref.get_entropy_loss()
                    loss += args.entropy_weight * entropy_loss

                # PG-only feature KD (feature MSE on upgraded channels only)
                if use_pg_kd and student_feats and teacher_feats and student_pg_blocks:
                    pg_kd_loss = 0.0
                    count = 0
                    for idx, student_block in enumerate(student_pg_blocks):
                        key = f'block_{idx}'
                        if key not in teacher_feats or key not in student_feats:
                            continue
                        s_feat = student_feats[key]
                        t_feat = teacher_feats[key]
                        if t_feat.size() != s_feat.size():
                            _, _, h, w = s_feat.size()
                            t_feat = F.interpolate(t_feat, size=(h, w), mode='bilinear', align_corners=False)
                        hard = student_block.conv2.get_hard_gates()
                        if hard is None:
                            continue
                        mask = hard.view(1, -1, 1, 1).to(s_feat.device)
                        diff = (s_feat - t_feat) * mask
                        pg_kd_loss = pg_kd_loss + torch.mean(diff * diff)
                        count += 1
                    if count > 0:
                        loss += args.pg_kd_weight * (pg_kd_loss / count)
            
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                ema_model.update(unwrap_model(net))

            # measure accuracy and record loss
            _, batch_predicted = torch.max(outputs.data, 1)
            batch_accu = 100.0 * (batch_predicted == hard_labels).sum().item() / hard_labels.size(0)
            losses.update(loss.item(), hard_labels.size(0))
            top1.update(batch_accu, hard_labels.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 99:
                # print statistics every 100 mini-batches each epoch
                progress.display(i)  # i = batch id in the epoch

        # update the learning rate
        scheduler.step()

        should_eval = should_run_eval(epoch)
        if should_eval:
            print('epoch {}'.format(epoch + 1))
            eval_model = ema_model.ema if ema_model is not None else net
            epoch_acc = test_accu(testloader, net, device, model_for_eval=eval_model)
            if 'pg' in _ARCH and should_run_interval(epoch, args.sparsity_interval):
                sparsity(testloader, eval_model, device)
            
            # Report adaptive PG gate statistics
            stats_model = eval_model if hasattr(eval_model, 'get_gate_statistics') else model_ref
            if 'adaptive-pg' in _ARCH and hasattr(stats_model, 'get_gate_statistics'):
                gate_stats = stats_model.get_gate_statistics()
                if gate_stats:
                    avg_active = np.mean([s['active_fraction'] for s in gate_stats])
                    print(f'Average 2-bit fraction: {avg_active:.3f} (target: {args.target_sparsity:.3f})')
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch + 1
                best_source = unwrap_model(eval_model) if hasattr(eval_model, 'state_dict') else model_ref
                best_model = copy.deepcopy(best_source.state_dict())
                print("New best model found.")
                if args.save:
                    print("Saving checkpoint.")
                    this_file_path = os.path.dirname(os.path.abspath(__file__))
                    save_folder = os.path.join(this_file_path, f'save_{args.dataset.upper()}_model')
                    suffix = _ARCH + '-finetune' if args.finetune else _ARCH
                    util.save_models(best_model, save_folder, suffix=suffix)
                    states = {'epoch': epoch + 1,
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}
                    util.save_states(states, save_folder, suffix=suffix)
            print("The best test accuracy so far: {:.1f} (epoch {})".format(best_acc, best_epoch))

    for handle in student_handles + teacher_handles:
        handle.remove()

    print('Finished Training')


#----------------------------
# Test accuracy.
#----------------------------
def test_accu(testloader, net, device, model_for_eval=None):
    correct = 0
    total = 0
    # switch the model to the evaluation mode
    eval_model = model_for_eval if model_for_eval is not None else net
    eval_model.eval()
    with torch.inference_mode():
        for data in testloader:
            images = data[0].to(device, non_blocking=True)
            labels = data[1].to(device, non_blocking=True)
            outputs = eval_model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print('Accuracy of the network on the %d test images: %.1f %%' % (total, accuracy))
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
    adaptive_only = any(
        isinstance(module, q.PGBinaryConv2d) and getattr(module, 'adaptive_pg', False)
        for module in net.modules()
    )
    total_out = 0.0
    total_high = 0.0
    with torch.inference_mode():
        for batch_idx, data in enumerate(testloader):
            images = data[0].to(device, non_blocking=True)
            _ = net(images)  # forward to populate counters
            outs, highs = collect_pg_counters()
            if outs:
                total_out += float(np.sum(outs))
            if highs:
                total_high += float(np.sum(highs))
            if adaptive_only:
                break
            if batch_idx >= 49:
                break

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
    model_ref = unwrap_model(net)
    if not ('adaptive-pg' in _ARCH and hasattr(model_ref, 'get_gate_statistics')):
        return
    
    print("\n" + "="*50)
    print("ADAPTIVE PG (Ada-FracBNN) ANALYSIS")
    print("="*50)
    
    # Gate statistics
    gate_stats = model_ref.get_gate_statistics()
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
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
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
            teacher_state = torch.load(args.teacher_path, map_location=device)
            if isinstance(teacher_state, dict) and 'state_dict' in teacher_state:
                teacher_state = teacher_state['state_dict']
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
            state_dict = torch.load(model_path, map_location=device)
            state_dict = remove_placeholder(state_dict)
            net.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Model not found.")

    #-----------------
    # Prepare Data
    #-----------------
    print("Loading the data.")
    trainloader, testloader = load_dataset()

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
        scheduler = build_scheduler(optimizer)
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
        scheduler = build_scheduler(optimizer)

        start_epoch = 0
        print("Start training.")
        train_model(trainloader, testloader, net,
                    optimizer, scheduler, start_epoch, device, teacher_net)
        final_acc = test_accu(testloader, net, device)
        
        # Analyze adaptive PG metrics
        analyze_adaptive_pg_metrics(net, testloader, device)


if __name__ == "__main__":
    main()
