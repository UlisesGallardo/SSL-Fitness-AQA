import torch
from pytorchvideo.accelerator.deployment.mobile_cpu.utils.model_conversion import (
    convert_to_deployable_form,
)
from pytorchvideo.models.accelerator.mobile_cpu.efficient_x3d import EfficientX3d
from models.pytorch_i3d import InceptionI3d



from torch.hub import load_state_dict_from_url
from torch.utils.mobile_optimizer import (
    optimize_for_mobile,
)


class FTModel(torch.nn.Module):
    def __init__(self, n_outputs=1):
        super().__init__()
        self.backbone = InceptionI3d()
            
        feature_dim = 1024 #1024
        self.head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, n_outputs)
        )


    def forward(self, x):
        x = self.backbone(x)
        x = torch.mean(x,-1)
        x = self.head(x)
        #x = self.getprob(x) #BCEwithlogits already has
        return x




#model_efficient_x3d_xs = EfficientX3d(expansion='XS', head_act='identity')
model = FTModel()

checkpoint_path = 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/efficient_x3d_xs_original_form.pyth'
checkpoint = load_state_dict_from_url(checkpoint_path)

#model_efficient_x3d_xs.load_state_dict(checkpoint)
path = "./checkpoint_supervied_ohp_k_bestf1.pt"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])

input_blob_size = (1, 3, 64, 220, 220)
input_tensor = torch.randn(input_blob_size)
model_efficient_x3d_xs_deploy = convert_to_deployable_form(model, input_tensor)
traced_model = torch.jit.trace(model_efficient_x3d_xs_deploy, input_tensor, strict=False)
optimized_traced__model = optimize_for_mobile(traced_model)
optimized_traced__model.save("./assets/video_classification.pt")
optimized_traced__model._save_for_lite_interpreter("./assets/video_classification.ptl")
