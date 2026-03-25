#!/usr/bin/env python3
"""
Train Full Precision ResNet-20 Teacher Model for Knowledge Distillation

This script trains a full-precision ResNet-20 model on CIFAR-10 that can be used
as a teacher for knowledge distillation with Ada-FracBNN student models.

Usage:
    python train_teacher.py -e 250 -b 128 -lr 0.1 -s
    python train_teacher.py --help  # for all options
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import utils.utils as util

import numpy as np
import os, time, sys
import copy
import argparse

#----------------------------
# Argument parser
#----------------------------
parser = argparse.ArgumentParser(description='Train Full Precision ResNet-20 Teacher for CIFAR-10')
parser.add_argument('--init_lr', '-lr', type=float, default=0.1,
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--batch_size', '-b', type=int, default=128,
                    help='batch size (default: 128)')
parser.add_argument('--num_epoch', '-e', type=int, default=250,
                    help='number of epochs (default: 250)')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--save', '-s', action='store_true',
                    help='save the trained model')
parser.add_argument('--test', '-t', action='store_true',
                    help='test only mode')
parser.add_argument('--resume', '-r', type=str, default=None,
                    help='resume from checkpoint')
parser.add_argument('--data_dir', '-d', type=str, default='/tmp/cifar10_data',
                    help='path to dataset directory')
parser.add_argument('--which_gpus', '-gpu', type=str, default='0',
                    help='which gpus to use')
parser.add_argument('--model_name', '-n', type=str, default='teacher',
                    help='model name for saving (default: teacher)')

args = parser.parse_args()

#----------------------------
# Load CIFAR-10 dataset
#----------------------------
def load_cifar10():
    """Load CIFAR-10 with ImageNet normalization (standard for full precision models)"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2,
                                              pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2,
                                             pin_memory=True)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

#----------------------------
# Generate teacher model
#----------------------------
def generate_teacher_model():
    """Generate full precision ResNet-20 teacher model"""
    import model.fracbnn_cifar10 as m
    return m.fp_resnet20(num_classes=10)

#----------------------------
# Train the teacher model
#----------------------------
def train_model(trainloader, testloader, net, optimizer, scheduler, device):
    """Train the full precision teacher model"""
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_model = copy.deepcopy(net.state_dict())
    
    print("\n" + "="*80)
    print("Starting Teacher Model Training")
    print("="*80)
    print(f"Model: Full Precision ResNet-20")
    print(f"Epochs: {args.num_epoch}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Initial LR: {args.init_lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print("="*80 + "\n")
    
    for epoch in range(args.num_epoch):
        # Training phase
        net.train()
        
        batch_time = util.AverageMeter('Time/batch', ':.2f')
        losses = util.AverageMeter('Loss', ':6.2f')
        top1 = util.AverageMeter('Acc', ':6.2f')
        progress = util.ProgressMeter(
            len(trainloader),
            [losses, top1, batch_time],
            prefix="Epoch: [{}]".format(epoch + 1)
        )
        
        print(f'Current learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        end = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # Measure accuracy
            _, batch_predicted = torch.max(outputs.data, 1)
            batch_accu = 100.0 * (batch_predicted == labels).sum().item() / labels.size(0)
            losses.update(loss.item(), labels.size(0))
            top1.update(batch_accu, labels.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % 100 == 99:
                progress.display(i)
        
        # Update learning rate
        scheduler.step()
        
        # Evaluation phase
        if epoch % 1 == 0:
            print(f'Epoch {epoch + 1}')
            epoch_acc = test_accuracy(testloader, net, device)
            
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(net.state_dict())
                
                if args.save:
                    this_file_path = os.path.dirname(os.path.abspath(__file__))
                    save_folder = os.path.join(this_file_path, 'save_teacher_model')
                    util.save_models(best_model, save_folder, suffix=args.model_name)
            
            print(f"Best test accuracy so far: {best_acc:.1f}%")
    
    print('\n' + "="*80)
    print('Finished Training Teacher Model')
    print(f'Best Test Accuracy: {best_acc:.1f}%')
    print("="*80)
    
    return best_acc

#----------------------------
# Test accuracy
#----------------------------
def test_accuracy(testloader, net, device):
    """Evaluate model on test set"""
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f'Accuracy on 10000 test images: {accuracy:.1f}%')
    return accuracy

#----------------------------
# Main function
#----------------------------
def main():
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Create teacher model
    print("\nCreating Full Precision ResNet-20 Teacher Model...")
    net = generate_teacher_model()
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print("Activating multi-GPU support")
        net = nn.DataParallel(net)
    net.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load checkpoint if resuming
    if args.resume is not None:
        model_path = args.resume
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            state_dict = torch.load(model_path)
            net.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"Model not found at {model_path}")
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    trainloader, testloader, classes = load_cifar10()
    print(f"Training samples: {len(trainloader.dataset)}")
    print(f"Test samples: {len(testloader.dataset)}")
    
    # Test only mode
    if args.test:
        print("\n" + "="*80)
        print("Test Only Mode")
        print("="*80)
        test_accuracy(testloader, net, device)
        return
    
    # Setup optimizer and scheduler
    print("\nSetting up optimizer and scheduler...")
    optimizer = optim.SGD(net.parameters(),
                         lr=args.init_lr,
                         momentum=0.9,
                         weight_decay=args.weight_decay)
    
    # Multi-step learning rate schedule (standard for CIFAR-10)
    milestones = [100, 150, 200]
    print(f"LR decay milestones: {milestones}")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=milestones,
                                               gamma=0.1)
    
    # Train the model
    final_acc = train_model(trainloader, testloader, net, 
                           optimizer, scheduler, device)
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    test_accuracy(testloader, net, device)
    
    # Save instructions
    if args.save:
        save_path = os.path.join(os.path.dirname(__file__), 
                                'save_teacher_model', 
                                f'model_{args.model_name}.pt')
        print("\n" + "="*80)
        print("Teacher Model Saved!")
        print("="*80)
        print(f"Location: {save_path}")
        print("\nTo use this teacher for Knowledge Distillation:")
        print(f"  python cifar10.py -id 2 -e 250 -b 128 -ts 0.15 -sw 0.01 \\")
        print(f"      -temp 4.0 -alpha 0.7 -tp {save_path} -s")
        print("="*80)

if __name__ == "__main__":
    main()

