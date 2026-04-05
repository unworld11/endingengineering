"""
A pre-activation variant of ReActNet as described in paper [2].
ReActNet is a binary model modified from MobileNet V1.

Reference:
    [1] MobileNet V1 code:
        https://github.com/marvis/pytorch-mobilenet
    [2] Author: Zechun Liu, Zhiqiang Shen, Marios Savvides,
                Kwang-Ting Cheng
        Title:  ReActNet: Towards Precise Binary Neural
                Network with Generalized Activation Functions
        URL:    https://arxiv.org/abs/2003.03488
"""


import sys
import time
import torch
import torch.nn as nn

sys.path.append('../')
import utils.quantization as q


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BuildingBlock(nn.Module):
    """
    Proposed ReActNet model variant.
    For details, please refer to our paper:
        https://arxiv.org/abs/2012.12206
    """
    def __init__(self, inp, oup, stride, adaptive_pg=False, target_sparsity=0.15):
        super(BuildingBlock, self).__init__()

        self.conv3x3 = nn.Sequential(
            q.PGBinaryConv2d(
                inp, inp, 3, stride, 1, bias=False,
                adaptive_pg=adaptive_pg,
                target_sparsity=target_sparsity,
            ),
            nn.BatchNorm2d(inp)
        )

        self.shortcut1 = nn.Sequential()
        if stride == 2:
            self.shortcut1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.pointwise = nn.Sequential(
            q.PGBinaryConv2d(
                inp, oup, 1, 1, 0, bias=False,
                adaptive_pg=adaptive_pg,
                target_sparsity=target_sparsity,
            ),
            nn.BatchNorm2d(oup)
        )

        self.shortcut2 = nn.Sequential()
        if oup == 2 * inp:
            self.shortcut2 = LambdaLayer(lambda x: torch.cat((x, x), dim=1))

        self.rprelu1 = q.RPReLU(in_channels=inp)
        self.rprelu2 = q.RPReLU(in_channels=oup)
        self.shiftbn1 = nn.BatchNorm2d(inp)
        self.shiftbn2 = nn.BatchNorm2d(oup)
        self.binarize = q.QuantSign.apply

    def forward(self, input):
        input = self.rprelu1(
            self.conv3x3(self.binarize(input))
        ) + self.shortcut1(self.binarize(input, 4))
        input = self.shiftbn1(input)
        input = self.rprelu2(
            self.pointwise(self.binarize(input))
        ) + self.shortcut2(self.binarize(input, 4))
        input = self.shiftbn2(input)
        return input


class ReActNet(nn.Module):
    def __init__(self, batch_size, num_gpus, adaptive_pg=False,
                 target_sparsity=0.15, num_classes=1000):
        super(ReActNet, self).__init__()
        self.adaptive_pg = adaptive_pg
        self.target_sparsity = target_sparsity
        self.num_classes = num_classes

        print("* FracBNN model.")
        print("* Precision gated activations.")
        print("* Binary input layer!")
        print("* Shortcuts are quantized to 4 bits.")
        if adaptive_pg:
            print("* Adaptive PG enabled.")

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                q.BinaryConv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
            )

        def conv_dw(inp, oup, stride):
            if inp == oup or 2 * inp == oup:
                return BuildingBlock(
                    inp, oup, stride,
                    adaptive_pg=adaptive_pg,
                    target_sparsity=target_sparsity,
                )
            raise NotImplementedError("Neither inp == oup nor 2*inp == oup")

        self.model = nn.Sequential(
            conv_bn(96, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, num_classes)

        assert batch_size % num_gpus == 0, \
            "Given batch size cannot evenly distributed to available gpus."
        n = batch_size // num_gpus
        self.encoder = q.InputEncoder(input_size=(n, 3, 224, 224), resolution=8)

        # ImageNet dataloaders in this codebase provide normalized inputs.
        # The binary encoder expects raw [0, 1] pixels, so the model internally
        # inverts normalization before thermometer encoding.
        self.register_buffer(
            'img_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            'img_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

    def get_sparsity_loss(self):
        total_loss = 0.0
        count = 0
        for module in self.modules():
            if isinstance(module, q.PGBinaryConv2d) and module.adaptive_pg:
                total_loss += module.get_sparsity_loss()
                count += 1
        return total_loss / count if count > 0 else 0.0

    def get_entropy_loss(self):
        total_loss = 0.0
        count = 0
        for module in self.modules():
            if isinstance(module, q.PGBinaryConv2d) and module.adaptive_pg:
                total_loss += module.get_entropy_loss()
                count += 1
        return total_loss / count if count > 0 else 0.0

    def set_temperature(self, temp):
        for module in self.modules():
            if isinstance(module, q.PGBinaryConv2d) and module.adaptive_pg:
                module.set_temperature(temp)

    def get_gate_statistics(self):
        stats = []
        for name, module in self.named_modules():
            if isinstance(module, q.PGBinaryConv2d) and module.adaptive_pg:
                layer_stats = module.get_gate_stats()
                layer_stats['layer_name'] = name
                stats.append(layer_stats)
        return stats

    def forward(self, x):
        x = x * self.img_std + self.img_mean
        x = self.encoder(x)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def speed(model, name):
    t0 = time.time()
    input = torch.rand(1, 3, 224, 224).cuda()
    t1 = time.time()

    model(input)
    t2 = time.time()

    model(input)
    t3 = time.time()

    print('%10s : %f' % (name, t3 - t2))


if __name__ == '__main__':
    reactnet = ReActNet(batch_size=1, num_gpus=1).cuda()
    speed(reactnet, 'reactnet')
