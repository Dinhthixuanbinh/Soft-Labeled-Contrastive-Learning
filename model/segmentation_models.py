# %%writefile /kaggle/working/Soft-Labeled-Contrastive-Learning/model/segmentation_models.py
import segmentation_models_pytorch as smp
from torch.nn import functional as F
from torch import nn
import torch
import numpy as np

# Assuming utils_ is importable if get_n_params is still used for prints
try:
    from utils.utils_ import get_n_params
except ImportError:
    def get_n_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


class segmentation_models(nn.Module):
    def __init__(self, name='resnet50', pretrained=False, 
                 in_channel=3, classes=4, 
                 decoder_channels=(256, 128, 64, 32, 16), # Provide default decoder_channels
                 multilvl=False, args=None): 
        super(segmentation_models, self).__init__()
        
        print(f"DEBUG: Initializing segmentation_models (ULTRA-SIMPLIFIED FOR TEST) with:")
        print(f"  name='{name}', pretrained={pretrained}, classes={classes}, in_channel={in_channel}")
        print(f"  decoder_channels for smp.Unet: {decoder_channels}")
        
        self.smp_model = smp.Unet(
            encoder_name=name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channel,
            classes=classes, 
            activation=None, # Get raw logits
            decoder_channels=decoder_channels 
        )
        print(f"DEBUG: smp.Unet model instantiated directly in segmentation_models.py.")
        # Store for compatibility with Trainer_baseline's prepare_model which might access self.encoder
        self.encoder = self.smp_model.encoder 
        print(f"Number of params (full smp.Unet model): {get_n_params(self.smp_model):,}")

        # Store args and multilvl for return signature compatibility for MCCL trainer if it uses this model
        self.args = args 
        self.multilvl = multilvl # From args, default False for baseline

    def forward(self, x, features_out=True): # features_out is for compatibility
        print(f"DEBUG: segmentation_models.forward (ULTRA-SIMPLIFIED) called with input shape {x.shape}")
        
        output_logits = self.smp_model(x) 
        # smp.Unet directly returns: (batch_size, num_classes, height, width)
        
        print(f"DEBUG: smp_model(x) output shape: {output_logits.shape}")

        # Trainer_baseline.train_epoch does: pred = out[0] if isinstance(out, tuple) else out
        # So returning a single tensor is fine for it.
        
        # To maintain a 3-part return for potential use by Trainer_MCCL (pred, bottleneck, dcdr_ft):
        if features_out or self.multilvl : # If MCCL trainer might be calling and expecting 3 outputs
            # This is a placeholder for bottleneck. A true bottleneck would be features[-1]
            bottleneck_placeholder = None 
            # Placeholder for decoder features (dcdr_ft). True dcdr_ft would be before smp_model's head
            dcdr_ft_placeholder = output_logits # Using logits as a stand-in
            return output_logits, bottleneck_placeholder, dcdr_ft_placeholder
        else: # For simple baseline that just wants the prediction
            return output_logits


# --- PointNet and segmentation_model_point ---
# Kept minimal for now, as train_baseline.py uses `segmentation_models`
class PointNet(nn.Module):
    def __init__(self, **kwargs): super().__init__(); # Minimal
    def forward(self, x): return x # Pass-through

class segmentation_model_point(segmentation_models):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs) # Pass all arguments to the (now simplified) base
        encoder_out_channels = self.smp_model.encoder.out_channels[-1]
        fc_inch = kwargs.get('fc_inch', 4)
        extpn = kwargs.get('extpn', False)
        self.pointnet = PointNet(num_points=300, fc_inch=fc_inch, conv_inch=encoder_out_channels, ext=extpn)
        print(f'Model {kwargs.get("name", "N/A")} with PointNet loaded (using simplified base).')

    def forward(self, x, features_out=True):
        # For this simplified test, segmentation_model_point will also just rely on base smp_model output
        segmentation_output = self.smp_model(x)
        
        # Placeholder for point features
        point_features = torch.randn((x.shape[0], 300, 3), device=x.device) if hasattr(self, 'pointnet') else None
        output_aux = None
        
        if self.multilvl:
            return segmentation_output, output_aux, point_features
        elif features_out:
            return segmentation_output, None, point_features # pred, bottleneck_placeholder, point_features
        else:
            return segmentation_output


if __name__ == '__main__':
    from torch import rand
    class DummyArgs: phead = False; multilvl = False # Minimal args for testing base
    
    print("\nTesting ULTRA-SIMPLIFIED base segmentation_models:")
    model_base = segmentation_models(name='resnet50', pretrained=False, classes=4, args=DummyArgs())
    
    img = rand((2, 3, 224, 224))
    output_tuple = model_base(img, features_out=True) # Call as trainer would with features_out=True
    
    if isinstance(output_tuple, tuple) and len(output_tuple) == 3:
        pred_base, bn_placeholder, ft_placeholder = output_tuple
        print("Simplified Base Model - Pred Output shape:", pred_base.shape)
        if bn_placeholder is not None: print("Simplified Base Model - Bottleneck placeholder shape:", bn_placeholder.shape)
        else: print("Simplified Base Model - Bottleneck placeholder is None")
        if ft_placeholder is not None: print("Simplified Base Model - Features placeholder shape:", ft_placeholder.shape)
    else: # Single tensor output
        print("Simplified Base Model (features_out=False or not tuple) - Pred Output shape:", output_tuple.shape)