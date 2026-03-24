from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import utils.quantization as q
import utils.utils as util


# Ignore "corrupt EXIF data" warnings in the console.
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


#########################
# supported model candidates

candidates = [
    'binput-pg-quant-shortcut',
    'adaptive-pg',
    'adaptive-pg-kd',
]
#########################


#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model_id', '-id', type=int, default=0)
parser.add_argument('--gtarget', '-g', type=float, default=0.0)
parser.add_argument('--init_lr', '-lr', type=float, default=5e-4)
parser.add_argument('--batch_size', '-b', type=int, default=256)
parser.add_argument('--num_epoch', '-e', type=int, default=120)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5)
parser.add_argument('--last_epoch', '-last', type=int, default=-1)
parser.add_argument('--finetune', '-f', action='store_true', help='finetune the model')
parser.add_argument('--save', '-s', action='store_true', help='save the model')
parser.add_argument('--test', '-t', action='store_true', help='test only')
parser.add_argument('--resume', '-r', type=str, default=None,
                    help='path of the model for resuming training')
parser.add_argument('--load_states', '-l', type=str, default=None,
                    help='path of states to the optimizer and scheduler')
parser.add_argument('--data_dir', '-d', type=str,
                    default='/temp/datasets/imagenet-pytorch/',
                    help='path to the dataset directory')
parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')
parser.add_argument('--workers', type=int, default=None,
                    help='number of dataloader workers (default: 2x visible GPUs, min 4)')
parser.add_argument('--target_sparsity', '-ts', type=float, default=0.15,
                    help='target 2-bit channel fraction for adaptive PG')
parser.add_argument('--sparsity_weight', '-sw', type=float, default=0.01,
                    help='weight for adaptive PG sparsity regularization')
parser.add_argument('--entropy_weight', '-ew', type=float, default=0.0,
                    help='weight for gate entropy regularization')
parser.add_argument('--kd_temperature', '-temp', type=float, default=4.0,
                    help='temperature for knowledge distillation')
parser.add_argument('--kd_alpha', '-alpha', type=float, default=0.7,
                    help='alpha for KD loss weighting')
parser.add_argument('--teacher_path', '-tp', type=str, default=None,
                    help='path to pretrained ImageNet teacher weights')
parser.add_argument('--teacher_arch', type=str, default='resnet50',
                    choices=['resnet18', 'resnet34', 'resnet50'],
                    help='teacher architecture for knowledge distillation')
parser.add_argument('--teacher_pretrained', action='store_true',
                    help='load torchvision ImageNet weights for the teacher')
parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='label smoothing for hard-label supervision')

args = parser.parse_args()

if args.model_id < 0 or args.model_id >= len(candidates):
    raise ValueError(
        f"Invalid model_id={args.model_id}. Must be in range [0, {len(candidates)-1}]. "
        f"Available models: {', '.join([f'{i}={c}' for i, c in enumerate(candidates)])}"
    )

_ARCH = candidates[args.model_id]
drop_last = False


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def resolve_num_workers():
    if args.workers is not None:
        return max(0, args.workers)
    return max(4, 2 * max(1, torch.cuda.device_count()))


def create_torchvision_model(model_name, pretrained=False):
    model_fn = getattr(torchvision.models, model_name)
    if not pretrained:
        try:
            return model_fn(weights=None)
        except TypeError:
            return model_fn(pretrained=False)

    weights_attr = f"{model_name.replace('resnet', 'ResNet')}_Weights"
    weights_enum = getattr(torchvision.models, weights_attr, None)
    if weights_enum is not None:
        return model_fn(weights=weights_enum.DEFAULT)
    return model_fn(pretrained=True)


#----------------------------
# Load the ImageNet dataset.
#----------------------------
def load_dataset():
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        util.Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transform=train_transforms
    )
    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transform=val_transforms
    )

    num_workers = resolve_num_workers()
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'drop_last': drop_last,
        'persistent_workers': num_workers > 0,
    }

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return trainloader, valloader


#----------------------------
# Define the model.
#----------------------------
def generate_model(model_arch):
    import model.fracbnn_imagenet as m

    adaptive_pg = 'adaptive-pg' in model_arch
    num_gpus = max(1, torch.cuda.device_count())
    return m.ReActNet(
        batch_size=args.batch_size,
        num_gpus=num_gpus,
        adaptive_pg=adaptive_pg,
        target_sparsity=args.target_sparsity,
    )


def generate_teacher_model():
    teacher = create_torchvision_model(
        args.teacher_arch,
        pretrained=args.teacher_pretrained,
    )
    return teacher


#----------------------------
# Train the network.
#----------------------------
def train_model(trainloader, testloader, net,
                optimizer, scheduler, start_epoch,
                num_epoch, device, teacher_net=None):
    criterion = q.LabelSmoothingCrossEntropy(
        label_smoothing=args.label_smoothing
    ).to(device)

    use_kd = teacher_net is not None
    if use_kd:
        kd_criterion = q.KnowledgeDistillationLoss(
            temperature=args.kd_temperature,
            alpha=args.kd_alpha,
            label_smoothing=args.label_smoothing,
        ).to(device)
        teacher_net.eval()
        print(f"Using Knowledge Distillation with T={args.kd_temperature}, alpha={args.kd_alpha}")

    best_acc = 0.0
    best_model = copy.deepcopy(unwrap_model(net).state_dict())
    states = {
        'epoch': start_epoch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    for epoch in range(start_epoch, num_epoch):
        batch_time = util.AverageMeter('Time/batch', ':.2f')
        losses = util.AverageMeter('Loss', ':6.2f')
        top1 = util.AverageMeter('Acc@1', ':6.2f')
        top5 = util.AverageMeter('Acc@5', ':6.2f')
        progress = util.ProgressMeter(
            len(trainloader),
            [losses, top1, top5, batch_time],
            prefix="Epoch: [{}]".format(epoch + 1)
        )

        net.train()
        print('current learning rate = {}'.format(optimizer.param_groups[0]['lr']))

        end = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            if use_kd:
                with torch.no_grad():
                    teacher_outputs = teacher_net(inputs)
                loss = kd_criterion(outputs, teacher_outputs, labels)
            else:
                loss = criterion(outputs, labels)

            if 'pg' in _ARCH:
                model_ref = unwrap_model(net)
                for name, param in model_ref.named_parameters():
                    if 'threshold' in name:
                        loss += (0.00001 * 0.5 *
                                 torch.norm(param - args.gtarget) *
                                 torch.norm(param - args.gtarget))

                if 'adaptive-pg' in _ARCH and hasattr(model_ref, 'get_sparsity_loss'):
                    loss += args.sparsity_weight * model_ref.get_sparsity_loss()

                if ('adaptive-pg' in _ARCH and hasattr(model_ref, 'get_entropy_loss')
                        and args.entropy_weight > 0.0):
                    loss += args.entropy_weight * model_ref.get_entropy_loss()

            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0].item(), inputs.size(0))
            top5.update(acc5[0].item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 99:
                progress.display(i)

        scheduler.step()

        model_ref = unwrap_model(net)
        if 'adaptive-pg' in _ARCH and hasattr(model_ref, 'set_temperature'):
            temp = 5.0 - 4.0 * (epoch / num_epoch)
            model_ref.set_temperature(max(temp, 1.0))

        print('epoch {}'.format(epoch + 1))
        epoch_acc = test_accu(testloader, net, device)
        if 'pg' in _ARCH:
            sparsity(testloader, net, device)

        if 'adaptive-pg' in _ARCH and hasattr(model_ref, 'get_gate_statistics'):
            gate_stats = model_ref.get_gate_statistics()
            if gate_stats:
                avg_active = np.mean([s['active_fraction'] for s in gate_stats])
                print(f'Average 2-bit fraction: {avg_active:.3f} (target: {args.target_sparsity:.3f})')

        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model_ref.state_dict())
            states = {
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
        print("Best test accuracy so far: {:.1f}".format(best_acc))

        if args.save:
            print("Saving the trained model.")
            this_file_path = os.path.dirname(os.path.abspath(__file__))
            save_folder = os.path.join(this_file_path, 'save_ImageNet_model')
            suffix = _ARCH + '-finetune' if args.finetune else _ARCH
            util.save_models(best_model, save_folder, suffix=suffix)
            util.save_states(states, save_folder, suffix=suffix)

    print('Finished Training')


#----------------------------
# Test accuracy.
#----------------------------
def accuracy(outputs, labels, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for
    the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test_accu(testloader, net, device):
    top1 = util.AverageMeter('Acc@1', ':6.2f')
    top5 = util.AverageMeter('Acc@5', ':6.2f')

    net.eval()

    with torch.no_grad():
        start = time.time()
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

        elapsed_time = time.time() - start
        print(' * Acc@1 {top1.avg:.1f} Acc@5 {top5.avg:.1f} Elapsed time = {clock:.1f} sec'
              .format(top1=top1, top5=top5, clock=elapsed_time))

    return top1.avg


#----------------------------
# Report sparsity in PG
#----------------------------
def sparsity(testloader, net, device):
    def collect_pg_counters():
        outs, highs = [], []

        def grab(m):
            if isinstance(m, q.PGBinaryConv2d):
                outs.append(np.asarray(getattr(m, 'num_out', 0)))
                highs.append(np.asarray(getattr(m, 'num_high', 0)))

        unwrap_model(net).apply(grab)
        return outs, highs

    unwrap_model(net).eval()
    total_out = 0.0
    total_high = 0.0
    batch_cnt = 50
    with torch.no_grad():
        start = time.time()
        for data in testloader:
            if batch_cnt == 0:
                break
            images, labels = data[0].to(device), data[1].to(device)
            _ = net(images)
            outs, highs = collect_pg_counters()
            if outs:
                total_out += float(np.sum(outs))
            if highs:
                total_high += float(np.sum(highs))
            batch_cnt -= 1
        elapsed_time = time.time() - start

    sparsity_pct = 100.0 - (total_high / total_out) * 100.0 if total_out > 0 else 0.0
    print('Sparsity of the update phase: {:.1f} %  Elapsed time = {clock:.1f} sec'.format(
          sparsity_pct, clock=elapsed_time))


#----------------------------
# Remove unused checkpoint keys
#----------------------------
def remove_placeholder(state_dict):
    from collections import OrderedDict
    temp_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if 'encoder.placeholder' in key:
            continue
        if 'teacher' in key:
            continue
        temp_state_dict[key] = value
    return temp_state_dict


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
        print("Activate multi GPU support.")
        net = nn.DataParallel(net)
    net.to(device)

    teacher_net = None
    if 'kd' in _ARCH:
        print("Create {} teacher.".format(args.teacher_arch))
        teacher_net = generate_teacher_model()
        if args.teacher_path is not None:
            if os.path.exists(args.teacher_path):
                print("@ Load teacher model from {}.".format(args.teacher_path))
                teacher_state = torch.load(args.teacher_path, map_location=device)
                if isinstance(teacher_state, dict) and 'state_dict' in teacher_state:
                    teacher_state = teacher_state['state_dict']
                teacher_net.load_state_dict(teacher_state, strict=False)
            else:
                raise ValueError("Teacher model not found.")
        elif not args.teacher_pretrained:
            print("WARNING: adaptive-pg-kd selected without --teacher_path or --teacher_pretrained.")
            print("Teacher will be randomly initialized, which is usually not useful for KD.")

        if torch.cuda.device_count() > 1:
            teacher_net = nn.DataParallel(teacher_net)
        teacher_net.to(device)
        teacher_net.eval()

    if args.resume is not None:
        model_path = args.resume
        if os.path.exists(model_path):
            print("@ Load trained model from {}.".format(model_path))
            state_dict = torch.load(model_path, map_location=device)
            state_dict = remove_placeholder(state_dict)
            net.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Model not found.")

    print("Loading the data.")
    trainloader, testloader = load_dataset()

    if args.test:
        print("Mode: Test only.")
        test_accu(testloader, net, device)
        if 'pg' in _ARCH:
            sparsity(testloader, net, device)
        return

    if args.finetune:
        print("num epochs = {}".format(args.num_epoch))
        initial_lr = args.init_lr
        print("init lr = {}".format(initial_lr))
        optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=0.0)

        print("Use linear learning rate decay.")
        lambda1 = lambda epoch: (1.0 - epoch / args.num_epoch)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda1,
            last_epoch=args.last_epoch,
        )
    else:
        print("num epochs = {}".format(args.num_epoch))
        initial_lr = args.init_lr
        print("init lr = {}".format(initial_lr))
        optimizer = optim.Adam(
            net.parameters(),
            lr=initial_lr,
            weight_decay=args.weight_decay,
        )

        print("Use linear learning rate decay.")
        lambda1 = lambda epoch: (1.0 - epoch / args.num_epoch)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda1,
            last_epoch=args.last_epoch,
        )

    start_epoch = 0
    if args.load_states is not None:
        states_path = args.load_states
        if os.path.exists(states_path):
            print("@ Load training states from {}.".format(states_path))
            states = torch.load(states_path, map_location=device)
            start_epoch = states['epoch']
            optimizer.load_state_dict(states['optimizer'])
            scheduler.load_state_dict(states['scheduler'])
        else:
            raise ValueError("Saved states not found.")

    print("Start {}.".format("finetuning" if args.finetune else "training"))
    train_model(
        trainloader, testloader, net,
        optimizer, scheduler, start_epoch,
        args.num_epoch, device, teacher_net=teacher_net
    )
    _ = test_accu(testloader, net, device)


if __name__ == "__main__":
    main()
