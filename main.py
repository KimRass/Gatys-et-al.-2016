# References:
    # https://github.com/Aleadinglight/Pytorch-VGG-19/blob/master/VGG_19.ipynb
    # https://deep-learning-study.tistory.com/680?category=983681
    # https://nuguziii.github.io/dev/dev-003/
    # https://nextjournal.com/gkoehler/pytorch-neural-style-transfer

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torchvision.models import vgg19, VGG19_Weights
from pathlib import Path
import argparse
import ssl
from copy import deepcopy
from tqdm import tqdm

import config
from utils import (
    get_device,
    load_image,
    save_image,
    resize,
    get_white_noise,
    FeatureMapExtractor,
    denorm,
)

ssl._create_default_https_context = ssl._create_unverified_context


def get_args():
    parser = argparse.ArgumentParser(description="Image Style Transfer")

    parser.add_argument("--content_image", required=True)
    parser.add_argument("--style_image", required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    # parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--alpha", type=int, required=False, default=1)
    parser.add_argument("--beta", type=int, required=False, default=1e9)

    args = parser.parse_args()
    return args


def get_images(args):
    content_image = load_image(args.content_image)
    style_image = load_image(args.style_image)

    content_image = resize(content_image, img_size=args.img_size)

    content_image = TF.to_tensor(content_image)
    content_image = TF.normalize(content_image, mean=config.MEAN, std=config.STD)

    style_image = TF.to_tensor(style_image)
    # "To extract image information on comparable scales, we always resized the style image
    # to the same size as the content image before computing its feature representations."
    _, h, w = content_image.shape
    style_image = TF.resize(style_image, size=(h, w), antialias=True)
    style_image = TF.normalize(style_image, mean=config.MEAN, std=config.STD)

    if config.FROM_CONTENT_IMAGE:
        gen_image = content_image.clone()
    else:
        gen_image = get_white_noise(content_image)

    content_image = content_image.unsqueeze(0)
    style_image = style_image.unsqueeze(0)
    gen_image = gen_image.unsqueeze(0)

    gen_image.requires_grad = True

    content_image = content_image.to(DEVICE)
    style_image = style_image.to(DEVICE)
    gen_image = gen_image.to(DEVICE)
    return content_image, style_image, gen_image


def _get_gram_mat(feat_map):
    _, c, _, _ = feat_map.shape
    # "A layer with $N_{l}$ distinct filters has $N_{l}$ feature maps each of size $M_{l}$, where $M_{l}$ is the height times the width of the feature map. So the responses in a layer $l$ can be stored in a matrix $F^{l} \in \mathbb{R}^{N_{l} \times M_{l}}$ where $F^{l}_{ij}$ is the activation of the $i$ th filter at position $j$ in layer $l$."
    # "$G^{l}_{ij} = \sum_{k}F^{l}_{ik}F^{l}_{jk}$"
    feat_map = feat_map.view((c, -1))
    gram_mat = torch.matmul(feat_map, torch.transpose(feat_map, dim0=0, dim1=1))
    return gram_mat


# "$w_{l}$"
def _get_contribution_of_layer(feat_map1, feat_map2):
    gram_mat1 = _get_gram_mat(feat_map1)
    gram_mat2 = _get_gram_mat(feat_map2)

    _, c, h, w = feat_map1.shape
    # "$E_{l} = \frac{1}{4N_{l}^{2}M_{l}^{2}} \sum_{i, j}\big(G^{l}_{x, ij} - G^{l}_{s, ij}\big)^{2}$"
    # contrib = F.mse_loss(gram_mat1, gram_mat2, reduction="mean") / (4 * (c * h * w) ** 2)
    contrib = F.mse_loss(gram_mat1, gram_mat2, reduction="sum") / (4 * (c * h * w) ** 2)
    return contrib


def freeze_feat_maps(feat_maps):
    for feat_map in feat_maps.values():
        feat_map.requires_grad = False


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def get_content_loss(gen_feat_maps, content_feat_maps):
    # "$L_{content}(\vec{x}, \vec{c}, l) = \frac{1}{2}\sum_{i, j}\big(F^{l}_{x, ij} - F^{l}_{c, ij}\big)^{2}$"
    content_loss = 0
    for layer_num in config.CONTENT_LAYER_NUMS:
        gen_feat_map = gen_feat_maps[layer_num]
        content_feat_map = content_feat_maps[layer_num].detach()
        content_loss += 0.5 * F.mse_loss(gen_feat_map, content_feat_map, reduction="sum")
    return content_loss


def get_style_loss(gen_feat_maps, style_feat_maps):
    style_loss = 0
    for weight, layer_num in zip(config.STYLE_WEIGHTS, config.STYLE_LAYERS_NUMS):
        gen_feat_map = gen_feat_maps[layer_num]
        style_feat_map = style_feat_maps[layer_num].detach()
        contrib = _get_contribution_of_layer(feat_map1=gen_feat_map, feat_map2=style_feat_map)
        style_loss += weight * contrib
    return style_loss


def maxpool2d_to_avgpool2d(model):
    # "For image synthesis we found that replacing the maximum pooling operation by average pooling yields
    # slightly more appealing results, which is why the images shown were generated with average pooling."
    for i, layer in enumerate(model.features):
        if isinstance(layer, torch.nn.MaxPool2d):
            model.features[i] = torch.nn.AvgPool2d(
                kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
            )


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()

    # "We used the feature space provided by a normalised version of the 16 convolutional and 5 pooling layers
    # of the 19-layer VGG network."
    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(DEVICE)
    
    maxpool2d_to_avgpool2d(model)
    model.eval()
    freeze_model(model)

    content_model = deepcopy(model)
    style_model = deepcopy(model)
    gen_model = deepcopy(model)

    content_image, style_image, gen_image = get_images(args)

    content_feat_maps = FeatureMapExtractor(model=content_model, layer_nums=config.LAYER_NUMS)(content_image)
    style_feat_maps = FeatureMapExtractor(model=style_model, layer_nums=config.LAYER_NUMS)(style_image)

    # "Here we use L-BFGS, which we found to work best for image synthesis."
    optim = Adam([gen_image], lr=config.LR)
    # scaler = GradScaler()

    exctractor = FeatureMapExtractor(model=gen_model, layer_nums=config.LAYER_NUMS)

    for epoch in tqdm(range(1, config.N_EPOCHS + 1)):
        # with torch.autocast(
        #     device_type=DEVICE.type,
        #     dtype=torch.float16,
        #     enabled=True if DEVICE.type == "cuda" else False,
        # ):
        gen_feat_maps = exctractor(gen_image)
        content_loss = get_content_loss(gen_feat_maps=gen_feat_maps, content_feat_maps=content_feat_maps)
        style_loss = get_style_loss(gen_feat_maps=gen_feat_maps, style_feat_maps=style_feat_maps)
        tot_loss = args.alpha * content_loss + args.beta * style_loss

        optim.zero_grad()
        tot_loss.backward()
        optim.step()

    image = gen_image.clone()
    image = image.detach().cpu()
    image = denorm(image, mean=config.MEAN, std=config.STD)
    image.clip_(0, 1)
    image = TF.to_pil_image(image.squeeze())
    save_image(
        image,
        path=Path(args.save_dir)/f"alpha_{int(args.alpha)}_beta_{int(args.beta)}.jpg",
    )
