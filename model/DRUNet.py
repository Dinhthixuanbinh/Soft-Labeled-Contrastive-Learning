import os.path

from torch import nn, cat
from torch import rand
import torch.nn.functional as F

from utils.utils_ import get_n_params


# implementation of DR_UNet


class Encoder(nn.Module):

    def __init__(self, filters=64, in_channels=3, n_block=3, kernel_size=(3, 3), batch_norm=True, padding='same'):
        super().__init__()
        self.filter = filters
        for i in range(n_block):
            out_ch = filters * 2 ** i
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = filters * 2 ** (i - 1)

            if padding == 'same':
                pad = kernel_size[0] // 2
            else:
                pad = 0
            model = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                     nn.LeakyReLU(inplace=True)]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            model += [nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                      nn.LeakyReLU(inplace=True)]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            self.add_module('encoder%d' % (i + 1), nn.Sequential(*model))
            conv = [nn.Conv2d(in_channels=in_ch * 3, out_channels=out_ch, kernel_size=1), nn.LeakyReLU(inplace=True)]
            self.add_module('conv1_%d' % (i + 1), nn.Sequential(*conv))

    def forward(self, x):
        skip = []
        output = x
        res = None
        i = 0
        for name, layer in self._modules.items():
            if i % 2 == 0:
                output = layer(output)
                skip.append(output)
            else:
                if i > 1:
                    output = cat([output, res], 1)
                    output = layer(output)
                output = nn.MaxPool2d(kernel_size=(2, 2))(output)
                res = output
            i += 1
        return output, skip


class Bottleneck(nn.Module):
    def __init__(self, filters=64, n_block=3, depth=4, kernel_size=(3, 3)):
        super().__init__()
        out_ch = filters * 2 ** n_block
        in_ch = filters * 2 ** (n_block - 1)
        for i in range(depth):
            dilate = 2 ** i
            model = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=dilate,
                               dilation=dilate), nn.LeakyReLU(inplace=True)]
            self.add_module('bottleneck%d' % (i + 1), nn.Sequential(*model))
            if i == 0:
                in_ch = out_ch

    def forward(self, x):
        bottleneck_output = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            bottleneck_output += output
        return bottleneck_output


class Decoder(nn.Module):
    def __init__(self, filters=64, n_block=4, kernel_size=(3, 3), batch_norm=True, padding='same', drop=False):
        super().__init__()
        self.n_block = n_block
        if padding == 'same':
            pad = kernel_size[0] // 2
        else:
            pad = 0
        for i in reversed(range(n_block)):
            out_ch = filters * 2 ** i
            in_ch = 2 * out_ch
            model = [nn.UpsamplingNearest2d(scale_factor=2),
                     nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                               padding=pad)]
            self.add_module('decoder1_%d' % (i + 1), nn.Sequential(*model))

            model = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                     nn.LeakyReLU(inplace=True)]
            if drop:
                model += [nn.Dropout(.5)]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            model += [nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                      nn.LeakyReLU(inplace=True)]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            self.add_module('decoder2_%d' % (i + 1), nn.Sequential(*model))

    def forward(self, x, skip, adaptseg=False):
        i = 0
        output = x
        output1 = None
        for _, layer in self._modules.items():
            output = layer(output)
            if adaptseg and i == len(self._modules.items()) - 3:
                output1 = output
            if i % 2 == 0:
                output = cat([skip.pop(), output], 1)
            i += 1
        if adaptseg:
            return output1, output
        else:
            return output


class Segmentation_model(nn.Module):
    def __init__(self, filters=32, in_channels=3, n_block=4, bottleneck_depth=4, n_class=4, feature_dis=True,
                 multilvl=False,
                 args=None):
        super().__init__()
        self.multilvl = multilvl
        self.encoder = Encoder(filters=filters, in_channels=in_channels, n_block=n_block)
        self.bottleneck = Bottleneck(filters=filters, n_block=n_block, depth=bottleneck_depth)
        self.decoder = Decoder(filters=filters, n_block=n_block)
        self.classifier = nn.Conv2d(in_channels=filters, out_channels=n_class, kernel_size=(1, 1))
        if multilvl:
            self.classifier1 = nn.Conv2d(in_channels=filters * 2, out_channels=n_class, kernel_size=(1, 1))
        # self.activation = nn.Softmax2d()
        if args is not None and 'phead' in vars(args):
            self.project_head = args.phead
        else:
            self.project_head = False
        if self.project_head:
            self.phead = nn.Sequential(*[nn.Conv2d(filters, filters * 2, kernel_size=1), nn.ReLU(),
                                         nn.Conv2d(filters * 2, filters, kernel_size=1)])
        self.number_params()

    def forward(self, x, features_out=True):
        # For input (N, 3, 256, 256)
        # output (N, 256, 16, 16), skip [(N, 32, 256, 256), (N, 64, 128, 128), # (N, 128, 64, 64), (N, 256, 32, 32)]
        output, skip = self.encoder(x)
        output_bottleneck = self.bottleneck(output)  # output_bottleneck (N, 512, 16, 16)
        if self.multilvl:
            output_decoder1, output_decoder = self.decoder(output_bottleneck, skip, adaptseg=True)
            output_decoder1 = F.interpolate(output_decoder1, size=x.size()[2:], mode='bilinear', align_corners=True)
            output_aux = self.classifier1(output_decoder1)  # output (N, 4, 256, 256)
        else:
            output_decoder = self.decoder(output_bottleneck, skip, adaptseg=False)  # output (N, 32, 256, 256)
        output = self.classifier(output_decoder)  # output (N, 4, 256, 256)

        if self.project_head:
            output_decoder = self.phead(output_decoder)
        if self.multilvl:
            return output, output_aux, output_decoder
        elif features_out:
            return output, output_bottleneck, output_decoder
        else:
            return output

    def number_params(self):
        print(f'Number of params: {get_n_params(self):,}')


if __name__ == '__main__':
    from utils.utils_ import write_model_graph

    img = rand((2, 3, 224, 224))
    model = Segmentation_model(multilvl=True)
    write_model_graph(model, img)
    # output = model.cuda()(img, features_out=False)
    # print(vert.size())
    # input()
    # from utils.utils_ import get_n_params
    # print(get_n_params(model)) # 13,483,844(filters=32, in_channels=3, n_block=4, bottleneck_depth=4, n_class=4, multilvl=False) |
    # 13,484,104(filters=32, in_channels=3, n_block=4, bottleneck_depth=4, n_class=4) | 544,676 (filters=16, n_blocks=3, bottleneck_depth=2, n_class=4)
    # 136,788(filters=8, n_blocks=3, bottleneck_depth=2, n_class=4) | 53,994,308 (filters=32, in_channels=3, n_block=5, bottleneck_depth=4)
    # 15,843,652 (filters=32, in_channels=3, n_block=4, bottleneck_depth=5) | 44,556,100(filters=32, in_channels=3, n_block=5, bottleneck_depth=3)
