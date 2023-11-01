import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from PIL import Image
import cv2
import requests
from io import BytesIO
from pathlib import Path
from copy import copy

import config


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def load_image(url_or_path):
    url_or_path = str(url_or_path)
    if "http" in url_or_path:
        response = requests.get(url_or_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(url_or_path).convert("RGB")
    return image


def downsample(img):
    return cv2.pyrDown(img)


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_pil(image).save(str(path), quality=100)


def get_white_noise(content_image):
    # "We jointly minimise the distance of the feature representations of a white noise image
    # from the content representation of the photograph in one layer and the style representation
    # of the painting defined on a number of layers of the Convolutional Neural Network."
    noise = torch.randn_like(content_image)
    noise = TF.normalize(noise, mean=config.MEAN, std=config.STD)
    return noise


def denorm(tensor, mean, std):
    tensor *= torch.Tensor(std)[None, :, None, None]
    tensor += torch.Tensor(mean)[None, :, None, None]
    return tensor


class FeatMapExtractor():
    def __init__(self, model, layer_nums):

        self.model = model

        self.feat_maps = dict()

        for layer_num in layer_nums:
            self.model.features[layer_num].register_forward_hook(self.get_feat_map(layer_num))

    def get_feat_map(self, layer_num):
        def forward_hook_fn(model, input, output):
            self.feat_maps[layer_num] = output
        return forward_hook_fn

    def __call__(self, image):
        self.model(image)
        return copy(self.feat_maps)


def resize(image, img_size):
    ori_w, ori_h = image.size
    if min(ori_w, ori_h) > img_size:
        if ori_w >= ori_h:
            w, h = round(ori_w * (img_size / ori_h)), img_size
        else:
            w, h = img_size, round(ori_h * (img_size / ori_w))
        new_image = image.resize(size=(w, h), )
        return new_image
    else:
        return image


def image_to_grid(content_image, style_image, gen_image):
    gen_image = gen_image.detach().cpu()
    image = torch.cat([content_image, style_image, gen_image], dim=0)
    image = denorm(image, mean=config.MEAN, std=config.STD)
    grid = make_grid(image, nrow=3, padding=2, pad_value=1)
    grid.clip_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid
