# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/model/segmentation_models.py
import segmentation_models_pytorch as smp
from torch.nn import functional as F
from torch import nn
import torch # Added torch import

from utils.utils_ import get_n_params


class segmentation_models(nn.Module):
    def __init__(self, name='resnet50', pretrained=False, decoder_channels=(512, 256, 128, 64, 32), in_channel=3,
                 classes=4, multilvl=False, args=None):
        super(segmentation_models, self).__init__()
        self.multilvl = multilvl
        self.classes = classes

        # Note: smp.Unet's 'classes' argument is for the final output layer if segmentation head is part of the model.
        # If you add your own classifier, you might set smp.Unet classes to a value that matches decoder_channels[-1]
        # or simply ensure your final self.classifier matches the decoder_channels[-1].
        # For simplicity and direct use of smp components:
        self.model = smp.Unet(
            encoder_name=name,
            encoder_weights='imagenet' if pretrained else None,
            decoder_channels=decoder_channels,
            in_channels=in_channel,
            classes=classes, # smp.Unet will have its own segmentation head
            activation=None, # Get raw logits from smp model
        )
        # To use your own custom head (classifier, phead, aux_classifier):
        # self.encoder = self.model.encoder
        # self.decoder = self.model.decoder
        # self.segmentation_head = self.model.segmentation_head # This is the smp head

        # If using custom head, then set classes in smp.Unet to 0 or decoder_channels[-1]
        # and build your own head. For now, let's assume we use smp's full Unet output.
        # The original code took model.encoder and model.decoder and built its own head.
        # Let's stick to that original structure for minimal changes to existing downstream code.

        _model_temp = smp.Unet( # Temporary model to extract encoder/decoder
            encoder_name=name,
            encoder_weights='imagenet' if pretrained else None, # This will load weights if pretrained=True
            decoder_channels=decoder_channels,
            in_channels=in_channel,
            classes=classes # Temporary, won't be used directly if using custom head
        )
        self.encoder = _model_temp.encoder
        self.decoder = _model_temp.decoder
        
        phead_in_channels = decoder_channels[-1]

        self.project_head = False
        self.phead = None
        if args is not None and hasattr(args, 'phead') and args.phead:
            self.project_head = True
            self.phead = nn.Sequential(
                nn.Conv2d(phead_in_channels, phead_in_channels * 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(phead_in_channels * 2, phead_in_channels, kernel_size=1)
            )

        self.classifier = nn.Conv2d(in_channels=phead_in_channels, out_channels=self.classes, kernel_size=(1, 1))
        
        self.classifier_aux = None
        if self.multilvl:
            if len(decoder_channels) >= 2:
                aux_in_channels = decoder_channels[-2]
                self.classifier_aux = nn.Conv2d(in_channels=aux_in_channels, out_channels=self.classes, kernel_size=(1, 1))
            else:
                print("Warning: multilvl=True but not enough decoder_channels to create auxiliary head. Disabling multilvl for aux head.")
                self.multilvl = False # Disable if not possible


        print(f'Model {name} loaded (encoder/decoder extracted).')
        # Note: get_n_params on self will now include only your custom head parts + encoder/decoder
        # If you want params of smp.Unet, then instantiate it fully and print params.
        print(f'Number of params (custom head + smp enc/dec): {get_n_params(self):,}')


    def forward(self, x, features_out=True):
        input_shape = x.shape[-2:]
        features = self.encoder(x) # This is a list of tensors

        # Deepest feature for the U-Net center
        center_feature = features[-1]
        decoder_output = self.decoder.center(center_feature)

        output_aux_features = None
        num_decoder_blocks = len(self.decoder.blocks)

        # SMP decoder blocks expect skips in order from shallowest relevant to deepest relevant for encoder
        # features = [input, stage1, stage2, stage3, stage4, stage5_deepest] (example for ResNet50, len=6)
        # Skips for decoder blocks (typically 4 blocks if 5 stages): features[4], features[3], features[2], features[1]
        # The smp.Decoder class handles this indexing internally based on encoder_depth.
        # We just need to pass all encoder features (except the one used for center and optionally input)
        # The smp.UnetDecoder takes features[1:] (all encoder outputs after initial stem/input)
        # and internally aligns them.
        
        # The decoder in SMP typically takes all features from the encoder
        # (often excluding the initial input image if the encoder handles the stem)
        # and the center block output.
        # Let's align with how smp.Unet's decoder would be called:
        # `self.decoder(*features)` implies features are passed such that skips are handled.
        # More explicitly, decoder blocks are iterated:
        
        # SMP's UnetDecoder.forward(self, *features) expects encoder features.
        # The structure is:
        # features = self.encoder(x)
        # decoder_output = self.decoder(*features) -> this then calls center and blocks
        # For our loop:
        
        # features[0] is usually input or stem output, features[-1] is deepest.
        # Skips are features[-2], features[-3], ..., features[1] (reversed order of depth)
        
        for i in range(num_decoder_blocks):
            skip_idx = len(features) - 2 - i # Correct index for skip connections from encoder features list
            skip_connection = features[skip_idx] if skip_idx >= 0 else None
            
            decoder_block = self.decoder.blocks[i]
            decoder_output = decoder_block(decoder_output, skip_connection)

            if self.multilvl and self.classifier_aux and (i == num_decoder_blocks - 2): # one block before the last
                output_aux_features = decoder_output
        
        # Main output
        final_decoder_output_for_cls = decoder_output
        output = self.classifier(final_decoder_output_for_cls)
        if output.shape[-2:] != input_shape:
            output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=True)

        # Aux output
        output_aux = None
        if self.multilvl and self.classifier_aux and output_aux_features is not None:
            output_aux = self.classifier_aux(output_aux_features)
            if output_aux.shape[-2:] != input_shape:
                output_aux = F.interpolate(output_aux, size=input_shape, mode='bilinear', align_corners=True)

        # Features for contrastive loss (from decoder_output before main classification)
        # Potentially pass through projection head
        features_for_contrastive = decoder_output
        if self.project_head and self.phead:
            projected_features = self.phead(features_for_contrastive)
        else:
            projected_features = features_for_contrastive


        if self.multilvl:
            return output, output_aux, projected_features
        elif features_out: # if not multilvl but features_out is True
            # Return main output, deepest encoder feature (bottleneck), and projected_features
            return output, features[-1], projected_features
        else: # Only main output
            return output


class PointNet(nn.Module):
    def __init__(self, num_points=300, fc_inch=4, conv_inch=2048, ext=False): # Default fc_inch to 4
        super().__init__()
        self.num_points = num_points
        self.ReLU = nn.LeakyReLU(inplace=True)
        if ext:
            self.conv1 = nn.Conv2d(conv_inch, conv_inch * 2, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(conv_inch * 2, conv_inch, kernel_size=3, padding=1)
        
        # Assuming input HxW=7x7 for ResNet50 features[-1] -> Conv6 -> 2x2 output -> Flattened=4
        # This fc_inch should match the flattened spatial dimension after final_conv
        print(f"PointNet: Initializing with fc_inch={fc_inch}. Ensure this matches flattened spatial dim after final_conv.")

        self.final_conv = nn.Conv2d(conv_inch, self.num_points, kernel_size=6) # Requires input H,W >= 6
        self.final_fc = nn.Linear(fc_inch, 3) 
        self._ext = ext

    def forward(self, x):
        if self._ext:
            x = self.ReLU(self.conv1(x))
            x = self.ReLU(self.conv2(x))
        x = self.ReLU(self.final_conv(x))
        current_size = x.size() # Should be (N, num_points, H_out, W_out)
        # Expected H_out * W_out should equal fc_inch for nn.Linear
        if current_size[2] * current_size[3] != self.final_fc.in_features:
            print(f"PointNet Warning: Flattened spatial dim {current_size[2]*current_size[3]} != fc_inch {self.final_fc.in_features}")
        x = x.view(current_size[0], current_size[1], -1) # Flatten spatial dimensions
        x = self.final_fc(x)
        return x


class segmentation_model_point(segmentation_models):
    def __init__(self, name='resnet50', pretrained=False, decoder_channels=(512, 256, 128, 64, 32), in_channel=3,
                 classes=4, multilvl=False, fc_inch=4, extpn=False, args=None): # Added args
        super(segmentation_model_point, self).__init__(name=name, pretrained=pretrained,
                                                       decoder_channels=decoder_channels, in_channel=in_channel,
                                                       classes=classes, multilvl=multilvl, args=args) # Pass args
        encoder_out_channels = self.encoder.out_channels[-1]
        self.pointnet = PointNet(num_points=300, fc_inch=fc_inch, conv_inch=encoder_out_channels, ext=extpn)
        print(f'Model {name} with PointNet loaded.')
        print(f'Number of params (Seg+PointNet): {get_n_params(self):,}')

    def forward(self, x, features_out=True): # features_out flag is for base class compatibility, not used here
        input_shape = x.shape[-2:]
        features = self.encoder(x)
        point = self.pointnet(features[-1])

        center_feature = features[-1]
        decoder_output = self.decoder.center(center_feature)
        
        num_decoder_blocks = len(self.decoder.blocks)
        output_aux_features = None

        for i in range(num_decoder_blocks):
            skip_idx = len(features) - 2 - i
            skip_connection = features[skip_idx] if skip_idx >= 0 else None
            decoder_block = self.decoder.blocks[i]
            decoder_output = decoder_block(decoder_output, skip_connection)

            if self.multilvl and self.classifier_aux and (i == num_decoder_blocks - 2):
                output_aux_features = decoder_output
        
        output = self.classifier(decoder_output)
        if output.shape[-2:] != input_shape:
            output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=True)

        output_aux = None
        if self.multilvl and self.classifier_aux and output_aux_features is not None:
            output_aux = self.classifier_aux(output_aux_features)
            if output_aux.shape[-2:] != input_shape:
                output_aux = F.interpolate(output_aux, size=input_shape, mode='bilinear', align_corners=True)
        
        return output, output_aux, point


if __name__ == '__main__':
    from torch import rand
    from utils.utils_ import write_model_graph

    # Example with dummy args for phead
    class DummyArgs:
        phead = True

    dummy_args = DummyArgs()

    img = rand((2, 3, 224, 224))
    model = segmentation_models(name='resnet50', pretrained=False, 
                                decoder_channels=(256, 128, 64, 32, 16), # Adjusted for 5 blocks
                                in_channel=3,
                                classes=4, multilvl=True, args=dummy_args)
    
    # Test the forward pass
    output, output_aux, projected_features = model(img)
    print("Output shape:", output.shape)
    if output_aux is not None:
        print("Aux output shape:", output_aux.shape)
    if projected_features is not None:
        print("Projected features shape:", projected_features.shape)
        
    # write_model_graph(model, img, '../runs/resnet50Mul_customHead_test') # Optional