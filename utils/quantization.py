import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


##########
##  Knowledge Distillation
##########

class KnowledgeDistillationLoss(nn.Module):
    """Knowledge Distillation Loss with temperature scaling"""
    def __init__(self, temperature=4.0, alpha=0.7):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss (scaled by temperature^2)
        kd_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss

def distillation_loss(student_output, teacher_output, labels, temperature=4.0, alpha=0.7):
    """Standalone knowledge distillation loss function"""
    kd_criterion = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)
    return kd_criterion(student_output, teacher_output, labels)


##########
##  ReAct
##########
'''
Implementations of react functions refer to:
    https://github.com/liuzechun/ReActNet
'''

class LearnableBias(nn.Module):
    def __init__(self, in_channels):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(
                        torch.zeros(1,in_channels,1,1), 
                        requires_grad=True
                    )

    def forward(self, input):
        return input + self.bias.expand_as(input)

class RPReLU(nn.Module):
    '''RPReLU is a PReLU sandwitched by learnable biases'''
    def __init__(self, in_channels):
        super(RPReLU, self).__init__()
        self.shift_x = LearnableBias(in_channels)
        self.shift_y = LearnableBias(in_channels)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, input):
        input = self.shift_x(input)
        input = self.prelu(input)
        input = self.shift_y(input)
        return input

class RSign(nn.Module):
    '''RSign is a Sign function that shifts the inputs'''
    def __init__(self, in_channels):
        super(RSign, self).__init__()
        self.shift_x = LearnableBias(in_channels)
        self.binarize = FastSign()

    def forward(self, input):
        input = self.shift_x(input)
        input = self.binarize(input)
        return input


##########
##  Quant
##########

class FastSign(nn.Module):
    def __init__(self):
        super(FastSign, self).__init__()

    def forward(self, input):
        out_forward = torch.sign(input)
        ''' 
        Only inputs in the range [-t_clip,t_clip] 
        have gradient 1. 
        '''
        t_clip = 1.3
        out_backward = torch.clamp(input, -t_clip, t_clip)
        return (out_forward.detach() 
                - out_backward.detach() + out_backward)

class QuantSign(torch.autograd.Function):
    '''
    Quantize Sign activation to arbitrary bitwidth.
    Usage: 
        output = QuantSign.apply(input, bits)
    '''
    @staticmethod
    def forward(ctx, input, bits=2):
        ctx.save_for_backward(input)
        input = torch.clamp(input, -1.0, 1.0)
        delta = 2.0/(2.0**bits-1.0)
        input = torch.round((input+1.0)/delta)*delta-1.0
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        ''' 
        Only inputs in the range [-t_clip,t_clip] 
        have gradient 1. 
        '''
        t_clip = 1.0
        grad_input = grad_output.clone()
        grad_input *= (input>-t_clip).float()
        grad_input *= (input<t_clip).float()
        return grad_input, None


##########
##  Mask
##########

class SparseGreaterThan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, torch.tensor(threshold))
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold, = ctx.saved_tensors
        grad_input = grad_output.clone()
        ''' Identity gradients only when input >= threshold '''
        grad_input *= (input>=threshold).float()
        return grad_input, None

class GreaterThan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        ''' Identity gradients '''
        grad_input = grad_output.clone()
        return grad_input, None


##########
##  Layer
##########

class BinaryConv2d(nn.Conv2d):
    '''
    A convolutional layer with its weight tensor binarized to {-1, +1}.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(BinaryConv2d, self).__init__(in_channels, out_channels,
                                              kernel_size, stride,
                                              padding, dilation, groups,
                                              bias, padding_mode)
        self.binarize = FastSign()

    def forward(self, input):
        return F.conv2d(input, self.binarize(self.weight),
                        self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

class PGBinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', sparse_bp=True, init=-1.0, 
                 adaptive_pg=False, target_sparsity=0.15):
        super(PGBinaryConv2d, self).__init__(in_channels, out_channels,
                                       kernel_size, stride,
                                       padding, dilation, groups,
                                       bias, padding_mode)
        self.binarize = FastSign()
        self.gt = SparseGreaterThan.apply if sparse_bp else GreaterThan.apply

        '''
        zero initialization
        nan loss while using torch.Tensor to initialize the thresholds
        '''
        self.threshold = nn.Parameter(torch.ones(1, out_channels, 1, 1)*init)

        ''' Adaptive PG: learnable gate vector '''
        self.adaptive_pg = adaptive_pg
        self.target_sparsity = target_sparsity
        if adaptive_pg:
            # Learnable gate vector g ∈ [0,1]^Cout (sigmoid)
            self.gate_logits = nn.Parameter(torch.randn(out_channels) * 0.1)
            # Temperature for annealing
            self.register_buffer('temperature', torch.tensor(1.0))
            # Threshold for hard decisions
            self.register_buffer('hard_threshold', torch.tensor(0.5))

        ''' number of output features '''
        self.num_out = torch.zeros(1)
        ''' number of output features computed at high precision '''
        self.num_high = torch.zeros(1)

    def forward(self, input):
        ''' MSB convolution '''
        out_msb = F.conv2d(self.binarize(input),
                           self.binarize(self.weight),
                           self.bias, self.stride, self.padding,
                           self.dilation, self.groups) * 2.0 / 3.0
        
        if self.adaptive_pg:
            # Adaptive PG: use learnable gates to determine which channels get 2-bit
            gates = torch.sigmoid(self.gate_logits / self.temperature)  # [out_channels]
            
            # Straight-through estimator for hard threshold
            hard_gates = (gates > self.hard_threshold).float()
            gates_ste = hard_gates.detach() + gates - gates.detach()
            
            # Expand gates to match output dimensions [1, out_channels, 1, 1]
            channel_mask = gates_ste.view(1, -1, 1, 1)
            
            ''' update report '''
            self.num_out.fill_( out_msb.numel() )
            self.num_high.fill_( (channel_mask > 0).sum().item() * out_msb.shape[2] * out_msb.shape[3] * out_msb.shape[0] )
            
            ''' full convolution only for selected channels '''
            out_full = F.conv2d(input,
                               self.binarize(self.weight),
                               self.bias, self.stride, self.padding,
                               self.dilation, self.groups)
            
            ''' combine outputs: channels with gate>threshold get full precision '''
            return (1-channel_mask) * out_msb + channel_mask * out_full
        else:
            # Original PG logic
            ''' Calculate the mask '''
            mask = self.gt(torch.sigmoid(5.0*(out_msb-self.threshold)), 0.5)
            ''' update report '''
            self.num_out.fill_( mask.numel() )
            self.num_high.fill_( (mask>0).sum().item() )
            ''' full convolution '''
            out_full = F.conv2d(input,
                               self.binarize(self.weight),
                               self.bias, self.stride, self.padding,
                               self.dilation, self.groups)
            ''' combine outputs '''
            return (1-mask) * out_msb + mask * out_full
    
    def get_sparsity_loss(self):
        """Return L1 penalty on gate vector for sparsity regularization"""
        if self.adaptive_pg:
            gates = torch.sigmoid(self.gate_logits)
            return torch.mean(gates)
        return 0.0
    
    def set_temperature(self, temp):
        """Set temperature for gate annealing"""
        if self.adaptive_pg:
            self.temperature.fill_(temp)
    
    def get_gate_stats(self):
        """Get statistics about gate activations"""
        if self.adaptive_pg:
            gates = torch.sigmoid(self.gate_logits)
            active_fraction = (gates > self.hard_threshold).float().mean().item()
            return {
                'active_fraction': active_fraction,
                'gate_mean': gates.mean().item(),
                'gate_std': gates.std().item()
            }
        return {}


##########
##  Transform
##########

class InputEncoder(nn.Module):
    '''
    Encode the input images to bipolar strings using thermometer encoding.
    Handles dynamic batch sizes during training.
    '''
    def __init__(self, input_size, resolution):
        super(InputEncoder, self).__init__()
        self.n, self.c, self.h, self.w = input_size
        self.resolution = int(resolution)
        self.b = int(round(255.0/self.resolution))
        
        # Create a template placeholder that can be expanded dynamically
        template = torch.arange(self.b, dtype=torch.float32).view(1, 1, -1, 1, 1)
        self.register_buffer('template', template)

    def forward(self, x):
        batch_size = x.size(0)
        x = (x * 255.0).view(batch_size, self.c, 1, self.h, self.w)
        
        # Expand template to match current batch size
        placeholder = self.template.expand(batch_size, self.c, self.b, self.h, self.w)
        
        output = (placeholder < torch.round(x/self.resolution)).float()
        output *= 2.0
        output -= 1.0
        return output.view(batch_size, self.b*self.c, self.h, self.w).detach()

