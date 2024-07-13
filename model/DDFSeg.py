from torch import nn
import torch
from torch.nn import functional as F


ngf = 32
def get_norm(out_channel, norm_type='Batch'):
    if norm_type is None:
        print("normalization type is not specified!")
        quit()
    elif norm_type == 'Ins':
        return nn.InstanceNorm2d(out_channel, affine=True)
    elif norm_type == 'Batch':
        return nn.BatchNorm2d(out_channel)


def get_relu(relufactor=.0):
    if relufactor == 0:
        return nn.ReLU(inplace=True)
    else:
        return nn.LeakyReLU(negative_slope=relufactor, inplace=True)


def general_conv2d(in_channel=3, out_channel=64, kernel_size=7, stride=1, stddev=0.01,
                   padding="same", do_norm=True, do_relu=True, zero_rate=None,
                   relufactor=0, norm_type='Batch'):
    """replacement of the function 'layers.general_conv2d' in the original tf version"""
    gen_conv2d = []
    m = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
    m.weight.data = torch.nn.init.trunc_normal_(m.weight.data, 0, stddev, -2, 2)
    if m.bias is not None:
        m.bias.data.zero_()
    gen_conv2d.append(m)
    if not zero_rate is None:
        gen_conv2d.append(nn.Dropout(zero_rate))
    if do_norm:
        gen_conv2d.append(get_norm(out_channel, norm_type=norm_type))
    if do_relu:
        gen_conv2d.append(get_relu(relufactor=relufactor))

    gen_conv2d = nn.Sequential(*gen_conv2d)

    return gen_conv2d


class Resnet_block(nn.Module):
    """replacement of the function 'model.build_resnet_block' in the original tf version"""
    def __init__(self, in_channel, out_channel, norm_type='Batch', zero_rate=0.25):
        super(Resnet_block, self).__init__()
        self.conv1 = general_conv2d(in_channel, out_channel, 3, 1, 0.01, "same", norm_type=norm_type,
                                    zero_rate=zero_rate)
        self.conv2 = general_conv2d(in_channel, out_channel, 3, 1, 0.01, "same", do_relu=False, norm_type=norm_type,
                                    zero_rate=zero_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + x)
        return out


class Resnet_block_ds(nn.Module):
    """replacement of the function 'model.build_resnet_block_ds' in the original tf version"""
    def __init__(self, in_channel, out_channel, norm_type='Batch', zero_rate=0.25):
        super(Resnet_block_ds, self).__init__()
        self.conv1 = general_conv2d(in_channel, out_channel, 3, 1, 0.01, "same", norm_type=norm_type,
                                    zero_rate=zero_rate)
        self.conv2 = general_conv2d(out_channel, out_channel, 3, 1, 0.01, "same", do_relu=False,
                                    norm_type=norm_type, zero_rate=zero_rate)
        self.relu = nn.ReLU(inplace=True)
        self.to_pad = (out_channel - in_channel) // 2

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        x = F.pad(x, (0, 0, 0, 0, self.to_pad, self.to_pad))
        return self.relu(out + x)


class Resnet_block_combine(nn.Module):
    def __init__(self, in_channel, out_channel, norm_type='Batch', zero_rate=0.25):
        super(Resnet_block_combine, self).__init__()
        self.res1 = Resnet_block_ds(in_channel, out_channel, norm_type, zero_rate)
        self.res2 = Resnet_block(out_channel, out_channel, norm_type, zero_rate)

    def forward(self, x):
        out = self.res2(self.res1(x))
        return out


class encoderc(nn.Module):
    """replacement of the function 'model.build_encoderc' in the original tf version"""
    def __init__(self, filters=16, zero_rate=None):
        super(encoderc, self).__init__()
        self.conv1 = general_conv2d(3, filters, 7, 1, 0.01, 'same', zero_rate=zero_rate, norm_type='Batch')
        self.res_block1 = Resnet_block(filters, filters, 'Batch', zero_rate)
        self.max_pool = nn.MaxPool2d(2)
        self.res_block_ds1 = Resnet_block_ds(filters, 2 * filters, 'Batch', zero_rate)
        self.res_block_combine = Resnet_block_combine(filters * 2, 4 * filters, 'Batch', zero_rate)
        self.res_block_combine1 = Resnet_block_combine(filters * 4, 8 * filters, 'Batch', zero_rate)
        self.res_block_combine2 = Resnet_block_combine(filters * 8, 16 * filters, 'Batch', zero_rate)
        self.res_block2 = Resnet_block(filters * 16, filters * 16, 'Batch', zero_rate)
        self.res_block3 = Resnet_block(filters * 16, filters * 16, 'Batch', zero_rate)
        self.res_block_combine3 = Resnet_block_combine(filters * 16, 32 * filters, 'Batch', zero_rate)

    def forward(self, x):
        out = self.conv1(x)  # (N, 16, 224, 224)
        out = self.res_block1(out)  # (N, 16, 224, 224)
        out = self.max_pool(out)  # (N, 16, 112, 112)
        out = self.res_block_ds1(out)  # (N, 32, 112, 112)
        out = self.max_pool(out)  # (N, 32, 56, 56)
        out = self.res_block_combine(out)  # (N, 64, 56, 56)
        out = self.max_pool(out)  # (N, 64, 28, 28)
        out = self.res_block_combine1(out)  # (N, 128, 28, 28)
        out = self.res_block_combine2(out)  # (N, 256, 28, 28)
        out = self.res_block2(out)  # (N, 256, 28, 28)
        out = self.res_block3(out)  # (N, 256, 28, 28)
        out = self.res_block_combine3(out)  # (N, 512, 28, 28)
        return out


def dilate_conv2d(in_channel=64, out_channel=64, kernel_size=7, dilation=2, stddev=0.01,
                  padding="same", do_norm=True, do_relu=True, zero_rate=None, relufactor=0, norm_type='Batch'):
    """replacement of the function 'layers.dilate_conv2d' in the original tf version"""
    dilate_conv = []
    m = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, dilation=dilation,
                  padding=padding)
    m.weight.data = torch.nn.init.trunc_normal_(m.weight.data, 0, stddev, -2, 2)
    if m.bias is not None:
        m.bias.data.zero_()
    dilate_conv.append(m)
    if not zero_rate is None:
        dilate_conv.append(nn.Dropout(p=zero_rate))

    if do_norm:
        dilate_conv.append(get_norm(out_channel, norm_type))
    if do_relu:
        dilate_conv.append(get_relu(relufactor))

    dilate_conv = nn.Sequential(*dilate_conv)

    return dilate_conv


class Attention_Module(nn.Module):
    """replacement of the function 'model.attention_2' in the original tf version"""
    def __init__(self, in_channel, out_channel, zero_rate=0.75):
        super(Attention_Module, self).__init__()
        self.conv1 = general_conv2d(in_channel, out_channel // 8, kernel_size=1, norm_type='Batch', zero_rate=zero_rate)
        self.max_pool = nn.MaxPool2d(2)
        self.conv2 = general_conv2d(in_channel, out_channel // 8, kernel_size=1, norm_type='Batch', zero_rate=zero_rate)
        self.conv3 = general_conv2d(in_channel, out_channel // 2, kernel_size=1, norm_type='Batch', zero_rate=zero_rate)
        self.gamma = torch.autograd.Variable(torch.tensor(0.0), requires_grad=True)
        self.conv4 = general_conv2d(out_channel // 2, out_channel, kernel_size=1, padding='valid', do_relu=False,
                                    norm_type='Batch', zero_rate=zero_rate)

    def forward(self, x):
        B, C, H, W = x.size()
        f = self.max_pool(self.conv1(x))
        g = self.conv2(x)
        h = self.max_pool(self.conv3(x))  # B * C // 2 * H * W
        f = f.flatten(2)  # B * C // 8 * N
        f = f.transpose(1, 2)  # B * N * C // 8
        g = g.flatten(2)
        s = torch.bmm(f, g)  # B * N * N
        beta = F.softmax(s, dim=1)
        h = h.flatten(2)  # B * C // 2 * N
        o = torch.bmm(h, beta)  # B * C // 2 * N
        o = o.reshape((B, C // 2, H, W))
        out = self.conv4(o)
        out = self.gamma * out + x
        return out


class Dilated_Resnet_Block(nn.Module):
    """replacement of the function 'model.build_drn_block' in the original tf version"""
    def __init__(self, channels, padding="same", norm_type=None, zero_rate=0.25):
        super(Dilated_Resnet_Block, self).__init__()
        self.dilated_conv1 = dilate_conv2d(in_channel=channels, out_channel=channels, kernel_size=3, dilation=2,
                                           stddev=0.01, padding=padding, do_relu=True, zero_rate=zero_rate,
                                           norm_type=norm_type)
        self.dilated_conv2 = dilate_conv2d(in_channel=channels, out_channel=channels, kernel_size=3, dilation=2,
                                           stddev=0.01, padding=padding, do_relu=False, zero_rate=zero_rate,
                                           norm_type=norm_type)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dilated_conv2(self.dilated_conv1(x))
        out = self.relu(out + x)
        return out


class encoders(nn.Module):
    """replacement of the function 'model.build_encoders' in the original tf version"""
    def __init__(self, filters=16, zero_rate=0.25):
        super(encoders, self).__init__()
        self.drn_block1 = Dilated_Resnet_Block(filters * 32, norm_type='Batch', zero_rate=zero_rate)
        self.drn_block2 = Dilated_Resnet_Block(filters * 32, norm_type='Batch', zero_rate=zero_rate)
        self.att = Attention_Module(filters * 32, filters * 32, zero_rate=zero_rate)

    def forward(self, x):
        out = self.drn_block1(x)
        out = self.drn_block2(out)
        out = self.att(out)
        return out  # N * 512 * ? * ?


class encoderdiff(nn.Module):
    """replacement of the function 'model.encoderdiffa' in the original tf version"""
    def __init__(self, filters=8, in_channel=3, kernel_size=3, zero_rate=.25):
        super(encoderdiff, self).__init__()
        self.conv1 = general_conv2d(in_channel, filters, 7, 1, 0.01, norm_type='Batch', zero_rate=zero_rate)
        self.res_block1 = Resnet_block(filters, filters, norm_type='Batch', zero_rate=zero_rate)
        self.max_pool = nn.MaxPool2d(2)
        self.res_block_ds1 = Resnet_block_ds(filters, filters * 2, 'Batch', zero_rate)
        self.res_block_ds2 = Resnet_block_ds(filters * 2, filters * 4, 'Batch', zero_rate)
        self.res_block2 = Resnet_block(filters * 4, filters * 4, 'Batch', zero_rate)
        self.conv2 = general_conv2d(filters * 4, 32, kernel_size, 1, 0.01, 'same', norm_type='Batch', zero_rate=zero_rate)
        self.conv3 = general_conv2d(32, 32, kernel_size, 1, 0.01, 'same', norm_type='Batch', zero_rate=zero_rate)

    def forward(self, x):
        out = self.conv1(x)  # (N, 8, 224, 224)
        out = self.res_block1(out)  # (N, 8, 224, 224)
        out = self.max_pool(out)  # (N, 8, 112, 112)
        out = self.res_block_ds1(out)  # (N, 16, 112, 112)
        out = self.max_pool(out)  # (N, 16, 56, 56)
        out = self.res_block_ds2(out)  # (N, 32, 56, 56)
        out = self.res_block2(out)  # (N, 32, 56, 56)
        out = self.max_pool(out)  # (N, 32, 28, 28)
        out = self.conv2(out)  # (N, 32, 28, 28)
        out = self.conv3(out)  # (N, 32, 28, 28)
        return out  # N * 32 * ? * ?


def general_deconv2d(in_channel, out_channel=64, kernel_size=7, stride=1, stddev=0.02, padding="valid",
                     do_norm=True, do_relu=True, relufactor=0, norm_type=None):
    """replacement of the function 'layers.general_deconv2d' in the original tf version"""
    gen_deconv = []
    m = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding=1 if padding == 'same' else 0,
                           output_padding=(1, 1) if padding == 'same' else 0)
    m.weight.data = torch.nn.init.trunc_normal_(m.weight.data, 0, stddev, -2, 2)
    if m.bias is not None:
        m.bias.data.zero_()
    gen_deconv.append(m)
    if do_norm:
        gen_deconv.append(get_norm(out_channel, norm_type=norm_type))
    if do_relu:
        gen_deconv.append(get_relu(relufactor=relufactor))
    gen_deconv = nn.Sequential(*gen_deconv)
    return gen_deconv

    
class decoderc(nn.Module):
    """replacement of the function 'model.decoderc' in the original tf version"""
    def __init__(self, in_channel=544, kernel_size=3):
        super(decoderc, self).__init__()
        self.conv1 = general_conv2d(in_channel, ngf * 4, kernel_size, stddev=0.02, padding='same', norm_type='Ins')
        self.res_block1 = Resnet_block(ngf * 4, ngf * 4, 'Ins')
        self.res_block2 = Resnet_block(ngf * 4, ngf * 4, 'Ins')
        self.res_block3 = Resnet_block(ngf * 4, ngf * 4, 'Ins')
        self.res_block4 = Resnet_block(ngf * 4, ngf * 4, 'Ins')

    def forward(self, x):
        out = self.conv1(x)  # (N, 128, 28, 28)
        out = self.res_block1(out)  # (N, 128, 28, 28)
        out = self.res_block2(out)  # (N, 128, 28, 28)
        out = self.res_block3(out)  # (N, 128, 28, 28)
        out = self.res_block4(out)  # (N, 128, 28, 28)
        return out  # N * 128 * ? * ?


class decodera(nn.Module):
    """replacement of the function 'model.decodera' and 'model.decoderb' in the original tf version"""
    def __init__(self, kernel_size=3, skip=False):
        super(decodera, self).__init__()
        self.decoder_block = decoderc(ngf * 4, kernel_size)
        self.deconv1 = general_deconv2d(ngf * 4, ngf * 2, kernel_size, 2, 0.02, 'same', norm_type='Ins')
        self.deconv2 = general_deconv2d(ngf * 2, ngf * 2, kernel_size, 2, 0.02, 'same', norm_type='Ins')
        self.deconv3 = general_deconv2d(ngf * 2, ngf, kernel_size, 2, 0.02, 'same', norm_type='Ins')
        self.conv1 = general_conv2d(ngf, 1, 7, 1, 0.02, 'same', False, False)
        self.skip = skip

    def forward(self, x, img):
        out = self.decoder_block(x)  # N * 128 * 28 * 28
        out = self.deconv1(out)  # (N, 64, 56, 56)
        out = self.deconv2(out)  # (N, 64, 112, 112)
        out = self.deconv3(out)  # (N, 32, 224, 224)
        out = self.conv1(out)  # (N, 1, 224, 224)
        if self.skip:
            out = out + img  # (N, 1, 224, 224)
        out = torch.tanh(out)  # (N, 1, 224, 224)
        return out


class DDFNet(nn.Module):
    def __init__(self):
        super(DDFNet, self).__init__()
        self.encoderc = encoderc()
        self.encoders = encoders()
        self.encodert = encoders()
        self.style_encoder_s = encoderdiff()
        self.style_encoder_t = encoderdiff()
        self.decoderc = decoderc()
        self.decoders = decodera(skip=True)
        self.decodert = decodera(skip=True)

    def content_encoder_s(self, x):
        return self.encoders(self.encoderc(x))

    def content_encoder_t(self, x):
        return self.encodert(self.encoderc(x))

    def decoder_s(self, x, img):
        return self.decoders(self.decoderc(x), img[:, 1: 2])

    def decoder_t(self, x, img):
        return self.decodert(self.decoderc(x), img[:, 1: 2])

    def forward(self, imgs, imgt):
        content_s = self.content_encoder_s(imgs)  # (N, 512, 28, 28)
        content_t = self.content_encoder_t(imgt)  # (N, 512, 28, 28)
        style_s = self.style_encoder_s(imgs)  # (N, 32, 28, 28)
        style_t = self.style_encoder_t(imgt)  # (N, 32, 28, 28)
        style_s_from_t = self.style_encoder_s(imgt)  # (N, 32, 28, 28) should be zero
        style_t_from_s = self.style_encoder_t(imgs)  # (N, 32, 28, 28) should be zero

        fake_img_s_t = self.decoder_t(torch.concat([content_s, style_t], dim=1), imgs)  # (N, 1, 224, 224)
        fake_img_t_s = self.decoder_s(torch.concat([content_t, style_s], dim=1), imgt)  # (N, 1, 224, 224)
        fake_img_s_t_ = torch.concat([fake_img_s_t, fake_img_s_t, fake_img_s_t], dim=1)  # (N, 3, 224, 224)
        fake_img_t_s_ = torch.concat([fake_img_t_s, fake_img_t_s, fake_img_t_s], dim=1)  # (N, 3, 224, 224)

        recon_content_t = self.content_encoder_s(fake_img_t_s_)  # (N, 512, 28, 28)
        recon_style_s = self.style_encoder_s(fake_img_t_s_)  # (N, 32, 28, 28)
        recon_content_s = self.content_encoder_t(fake_img_s_t_)  # (N, 512, 28, 28)
        recon_style_t = self.style_encoder_t(fake_img_s_t_)  # (N, 32, 28, 28)

        recon_imgs = self.decoder_s(torch.concat([recon_content_s, recon_style_s], dim=1), fake_img_s_t_)  # (N, 1, 224, 224)
        recon_imgt = self.decoder_t(torch.concat([recon_content_t, recon_style_t], dim=1), fake_img_t_s_)  # (N, 1, 224, 224)

        out = {'style_s_from_t': style_s_from_t, 'style_t_from_s': style_t_from_s,
               'fake_img_s_t': fake_img_s_t, 'fake_img_t_s': fake_img_t_s,
               'recon_imgs': recon_imgs,  'recon_imgt': recon_imgt, 'recon_content_s': recon_content_s,
               'content_t': content_t, 'content_s': content_s}
        return out


class SegDecoder(nn.Module):
    def __init__(self, zero_rate=0.25):
        super(SegDecoder, self).__init__()
        f = 7
        kernel_size = 3
        self.conv1 = general_conv2d(512, ngf * 4, kernel_size, 1, 0.02, norm_type='Ins', zero_rate=zero_rate)
        self.res_block1 = Resnet_block(ngf * 4, ngf * 4, 'Ins')
        self.res_block2 = Resnet_block(ngf * 4, ngf * 4, 'Ins')
        self.res_block3 = Resnet_block(ngf * 4, ngf * 4, 'Ins')
        self.res_block4 = Resnet_block(ngf * 4, ngf * 4, 'Ins')
        self.deconv1 = general_deconv2d(ngf * 4, ngf * 2, kernel_size, 2, 0.02, 'same', norm_type='Ins')
        self.deconv2 = general_deconv2d(ngf * 2, ngf * 2, kernel_size, 2, 0.02, 'same', norm_type='Ins')
        self.deconv3 = general_deconv2d(ngf * 2, ngf, kernel_size, 2, 0.02, 'same', norm_type='Ins')
        self.conv2 = general_conv2d(ngf, 4, f, 1, 0.02, do_relu=False, do_norm=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.conv2(out)
        return out


if __name__ == '__main__':
    imgc = torch.rand(2, 3, 224, 224).cuda()
    imgt = torch.rand(2, 3, 224, 224).cuda()
    # ddfnet = DDFNet().cuda()
    # out = ddfnet(imgc, imgt)
    segmentor = SegDecoder().cuda()
    out = segmentor(torch.rand(2, 512, 28, 28).cuda())
    print('program finished.')



