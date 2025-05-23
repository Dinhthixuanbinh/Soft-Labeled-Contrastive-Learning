import torch.nn as nn
import torch
from utils.utils_ import calc_mean_std, weights_init_kaiming, calc_feat_mean_std
from utils.utils_ import adaptive_instance_normalization_with_noise as adainwn
from utils.timer import timeit


def get_decoder():
    decoder = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )
    return decoder


def get_encoder():
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(inplace=True),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(inplace=True),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(inplace=True),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(inplace=True),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(inplace=True),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(inplace=True),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(inplace=True),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(inplace=True),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(inplace=True),  # relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(inplace=True),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(inplace=True),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(inplace=True),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(inplace=True),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(inplace=True),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(inplace=True),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(inplace=True)  # relu5-4
    )
    return encoder


def get_fc_encoder():
    fc_encoder = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024)
    )
    return fc_encoder


def get_fc_decoder():
    fc_decoder = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024)
    )
    return fc_decoder


@timeit
def load_rain_models(encoder_weight=None, decoder_weights=None, fc_encoder_weights=None, fc_decoder_weights=None,
                     device='cuda'):
    vgg_encoder, vgg_decoder, style_encoder, style_decoder = get_encoder().eval(), get_decoder().eval(), \
                                                             get_fc_encoder().eval(), get_fc_decoder().eval()
    # load pretrained weights for RAIN
    if encoder_weight is not None:
        vgg_encoder.load_state_dict(torch.load(encoder_weight))
        vgg_encoder = nn.Sequential(*list(vgg_encoder.children())[:31])
        print('vgg encoder loaded')
    if decoder_weights is not None:
        try:
            vgg_decoder.load_state_dict(torch.load(decoder_weights))
            print("decoder load from model")
        except:
            vgg_decoder.load_state_dict(torch.load(decoder_weights)['model_state_dict'])
            print("decoder load from state dict")
    if fc_encoder_weights is not None:
        try:
            style_encoder.load_state_dict(torch.load(fc_encoder_weights))
            print("fc_decoder load from model")
        except:
            style_encoder.load_state_dict(torch.load(fc_encoder_weights)['model_state_dict'])
            print("fc_decoder load from state dict")
    if fc_decoder_weights is not None:
        try:
            style_decoder.load_state_dict(torch.load(fc_decoder_weights))
            print("fc_encoder load from model")
        except:
            style_decoder.load_state_dict(torch.load(fc_decoder_weights)['model_state_dict'])
            print("fc_encoder load from state dict")
    for param in vgg_encoder.parameters():
        param.requires_grad = False
    return vgg_encoder.to(device), vgg_decoder.to(device), style_encoder.to(device), style_decoder.to(device)


class Net(nn.Module):
    def __init__(self, encoder, decoder, fc_encoder, fc_decoder, init=True):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.fc_encoder = fc_encoder
        self.fc_decoder = fc_decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        if init:
            self.decoder.apply(weights_init_kaiming)
            self.fc_encoder.apply(weights_init_kaiming)
            self.fc_decoder.apply(weights_init_kaiming)

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

        # extract relu4_1 from input image

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def calc_latent_loss(self, z_mean, z_stddev, eps=1e-5):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        # KL divergence between P = N(z_mean, z_stddev) and Q = N(0 ,1).
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq + eps) - 1)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        # print("s", style_feats.size())
        content_feat = self.encode(content)
        # content_feat = self.encode_with_intermediate(content)

        style_feat_mean_std = calc_feat_mean_std(style_feats[-1])

        intermediate = self.fc_encoder(style_feat_mean_std)
        intermediate_mean = intermediate[:, :512]
        intermediate_std = intermediate[:, 512:]
        noise = torch.randn_like(intermediate_mean)
        sampling = intermediate_mean + noise * intermediate_std  # N, 512
        style_feat_mean_std_recons = self.fc_decoder(sampling)  # N, 1024

        t = adainwn(content_feat, style_feat_mean_std_recons)
        # t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t.detach())
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0].detach())
        loss_l = self.calc_latent_loss(intermediate_mean, intermediate_std)
        loss_r = self.mse_loss(style_feat_mean_std_recons, style_feat_mean_std.detach())
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s, loss_l, loss_r

    def style_transfer(self, content, style, sampling=None):
        """
        RAIN implementation that generate images which preserve content of content images and style of style images
        Args:
            encoder: the VGG encoder
            decoder: the VGG-like decoder
            fc_encoder: the VAE encoder
            fc_decoder: the VAE decoder
            content: the content images
            style: the style images
            sampling: the epsilon sampled from a distribution generated by the VAE encoder
            eps: Whether to sample the style feature based on the mean and std of the feature

        Returns:
        Images which preserve content of content images and style of style images, the sampling which will be updated for
        the following iterations
        """
        with torch.no_grad():
            content_feat = self.encode(content)  # (N, 512, 28, 28)
            style_feat = self.encode(style)  # (N, 512, 28, 28)
        if sampling is None:
            style_feat_mean_std = calc_feat_mean_std(style_feat)  # (N, 1024)
            intermediate = self.fc_encoder(style_feat_mean_std)  # (N, 1024)
            intermediate_mean = intermediate[:, :512]  # (N, 512)
            intermediate_std = intermediate[:, 512:]  # (N, 512)
            noise = torch.randn_like(intermediate_mean)
            sampling = intermediate_mean + noise * intermediate_std  # (N, 512) sample the feature of the style
        # sampling.requires_grad = True
        style_feat_mean_std_recons = self.fc_decoder(sampling)  # (N, 1024)
        feat = adainwn(content_feat, style_feat_mean_std_recons)  # (N, 512, 28, 28)
        # feat = []
        # for style_statistics in style_feat_mean_std_recons:
        #     feat.append(adainwn(content_feat, torch.unsqueeze(style_statistics, 0)))
        # feat = torch.concat(feat, 0)

        return self.decoder(feat), sampling


if __name__ == '__main__':
    from torch import rand
    import numpy as np

    img = rand((2, 3, 224, 224)).cuda()
