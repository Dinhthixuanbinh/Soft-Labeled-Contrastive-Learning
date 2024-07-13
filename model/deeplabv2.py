import torch
import torch.nn as nn

from utils.utils_ import get_n_params

affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # change
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        # change
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, multi_level=False, input_size=224):
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        if self.multi_level:
            self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.interp = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=True)
        self.number_params()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.multi_level:
            output_aux = self.layer5(x)  # produce segmap 1 # inchannel = 1024, out_channel = num_classes
            output_aux = self.interp(output_aux)
        else:
            output_aux = x
        out_ft = self.layer4(x)  # inchannel= 1024, out_channel = 2048  x2 class feature
        output = self.layer6(out_ft)  # produce segmap 2 inchannel_2048 out_channel = num_classes
        output = self.interp(output)
        return output, output_aux, out_ft

    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].parameters():
                if j.requires_grad:
                    yield j

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]

    def number_params(self):
        print(f'Number of params: {get_n_params(self):,}')


def get_deeplab_v2(num_classes=19, layers=(3, 4, 23, 3), multi_level=True, input_size=224):
    model = ResNetMulti(Bottleneck, layers, num_classes, multi_level, input_size=input_size)
    return model


def write_model_graph(model, images=None, log_dir='../runs/model_graph'):
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    import os
    from pathlib import Path
    if not Path(log_dir).parent.exists():
        Path(log_dir).parent.mkdir(parents=True)
    if os.path.exists(log_dir):
        now = datetime.now()
        log_dir += f'.{now.hour}.{now.minute}'
    writer = SummaryWriter(log_dir=log_dir)
    if images is None:
        images = torch.rand(2, 3, 224, 224)
    writer.add_graph(model, images)
    writer.close()
    print(f'graph saved at: {Path(writer.log_dir).absolute()}')


if __name__ == "__main__":
    # model = get_deeplab_v2(4, layers=(3, 4, 23, 3), multi_level=True)
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.encoders import get_preprocessing_fn
    # resnet101 51,513,668 | resnet34 24,436,659 | resnet50 32,521,540 | resnext50_32x4d 31,993,412 | timm-resnest50d 34,447,748
    # timm-efficientnet-b6 49,270,236
    model_name = 'deeplabv3'
    model = smp.Unet(
        encoder_name=model_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,
        decoder_channels=[512, 256, 128, 64, 32],
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=4,  # model output channels (number of classes in your dataset)
    )
    print(f'num_params: {get_n_params(model):,}')
    img = torch.rand(2, 3, 224, 224)
    write_model_graph(model, img, f'../runs/{model_name}')
    # out = model(img)
    print('finish')
    # write_model_graph(model, log_dir='../runs/ResNetMulti')
    # 42,942,560(num_classes=4, multi_level=True) | 42,795,088(num_classes=4, multi_level=False) |
    # 30,309,216(num_classes=4, layers=(2, 3, 16, 2), multi_level=True) | 23,606,112(num_classes=4, layers=(2, 3, 10, 2), multi_level=True) |
    # 18,020,192(num_classes=4, layers=(2, 3, 5, 2), multi_level=True) | 15,505,760(num_classes=4, layers=(2, 2, 3, 2), multi_level=True)
