import torch.nn as nn
import torchvision.transforms as T
import numpy as np


def print_all_layers(model):
    for name, module in model.named_modules():
        if isinstance(
            module,
            (nn.Linear, nn.Conv2d, nn.MaxPool2d, nn.AdaptiveMaxPool2d, nn.ReLU)
        ):
            print(f"""| {name:20s}: {str(type(module)):50s} |""")


def _denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    copied_img = img.copy()
    copied_img *= std
    copied_img += mean
    copied_img *= 255.0
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


def tensor_to_array(tensor):
    copied_tensor = tensor.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    copied_tensor = _denormalize(copied_tensor)
    return copied_tensor


def _get_target_layer(layer):
    return eval(
        "model" + "".join(
            [f"""[{i}]""" if i.isdigit() else f""".{i}""" for i in layer.split(".")]
        )
    )


class FeatureMapExtractor():
    def __init__(self, model):
        self.model = model

        self.feat_map = None

    def get_feature_map(self, image, layer):
        def forward_hook_fn(module, input, output):
            self.feat_map = output

        trg_layer = _get_target_layer(layer)
        trg_layer.register_forward_hook(forward_hook_fn)

        self.model(image)
        return self.feat_map


def get_content_image_transformer():
    style_transform = T.Compose(
        [
            T.ToTensor(),
            # T.CenterCrop(224),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    return style_transform


# "To extract image information on comparable scales, we always resized the style image
# to the same size as the content image before computing its feature representations."
def get_style_image_transformer(content_img):
    h, w, _ = content_img.shape
    style_transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((h, w)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    return style_transform
