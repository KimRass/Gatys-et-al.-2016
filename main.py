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
from tqdm import tqdm
import ssl

import config
from utils import (
    get_device,
    load_image,
    save_image,
    resize,
    get_white_noise,
    FeatMapExtractor,
    image_to_grid
)

ssl._create_default_https_context = ssl._create_unverified_context


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--content_img", required=True)
    parser.add_argument("--style_img", required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--alpha", type=int, required=False, default=1)
    parser.add_argument("--beta", type=int, required=False, default=1e8)

    args = parser.parse_args()
    return args


def get_images(args):
    ori_content_image = load_image(args.content_img)
    content_image = resize(ori_content_image, img_size=config.IMG_SIZE)

    ori_content_image = TF.to_tensor(ori_content_image)
    ori_content_image = TF.normalize(ori_content_image, mean=config.MEAN, std=config.STD)

    content_image = TF.to_tensor(content_image)
    content_image = TF.normalize(content_image, mean=config.MEAN, std=config.STD)

    style_image = load_image(args.style_img)
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

    ori_content_image = ori_content_image.unsqueeze(0)
    content_image = content_image.unsqueeze(0)
    style_image = style_image.unsqueeze(0)
    gen_image = gen_image.unsqueeze(0)

    gen_image.requires_grad = True

    content_image = content_image.to(DEVICE)
    style_image = style_image.to(DEVICE)
    gen_image = gen_image.to(DEVICE)
    return ori_content_image, content_image, style_image, gen_image


def _get_gram_mat(feat_map):
    _, c, _, _ = feat_map.shape
    # "A layer with $N_{l}$ distinct filters has $N_{l}$ feature maps each of size $M_{l}$, where $M_{l}$ is
    # the height times the width of the feature map. So the responses in a layer $l$ can be stored in a matrix
    # $F^{l} \in \mathbb{R}^{N_{l} \times M_{l}}$ where $F^{l}_{ij}$ is the activation of the $i$ th filter at
    # position $j$ in layer $l$."
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
    freeze_model(model)
    model.eval()

    ori_content_image, content_image, style_image, gen_image = get_images(args)

    exctractor = FeatMapExtractor(model=model, layer_nums=config.LAYER_NUMS)
    content_feat_maps = exctractor(content_image)
    style_feat_maps = exctractor(style_image)

    # "Here we use L-BFGS, which we found to work best for image synthesis."
    # 논문에서와 다르게 Adam optimizer를 사용하겠습니다. 학습 속도에 있어서 이쪽이 훨씬 빠른 것 같습니다.
    optim = Adam([gen_image], lr=config.LR, betas=(config.BETA1, config.BETA2), eps=config.EPS)

    if DEVICE.type == "cuda":
        scaler = GradScaler()

    for epoch in tqdm(range(1, config.N_EPOCHS + 1)):
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16,
            enabled=True if DEVICE.type == "cuda" else False,
        ):
            gen_feat_maps = exctractor(gen_image)
            content_loss = get_content_loss(gen_feat_maps=gen_feat_maps, content_feat_maps=content_feat_maps)
            style_loss = get_style_loss(gen_feat_maps=gen_feat_maps, style_feat_maps=style_feat_maps)
            tot_loss = args.alpha * content_loss + args.beta * style_loss

        if DEVICE.type == "cuda":
            optim.zero_grad()
            scaler.scale(tot_loss).backward()
            scaler.step(optim)
        else:
            optim.zero_grad()
            tot_loss.backward()
            optim.step()

    _, _, ori_h, ori_w = ori_content_image.shape
    style_image = TF.resize(style_image, size=(ori_h, ori_w), antialias=True)
    gen_image = TF.resize(gen_image, size=(ori_h, ori_w), antialias=True)
    grid = image_to_grid(content_image=ori_content_image, style_image=style_image, gen_image=gen_image)
    save_image(
        grid,
        path=Path(args.save_dir)/f"{Path(args.content_img).stem}_{Path(args.style_img).stem}.jpg",
    )
