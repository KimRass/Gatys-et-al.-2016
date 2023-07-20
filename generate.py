# References:
    # https://github.com/Aleadinglight/Pytorch-VGG-19/blob/master/VGG_19.ipynb
    # https://deep-learning-study.tistory.com/680?category=983681

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg19_bn, VGG19_BN_Weights
import numpy as np
from pathlib import Path

from utils import get_args
from image_utils import load_image, save_image, downsample, get_white_noise
from torch_utils import (
    tensor_to_array,
    print_all_layers,
    FeatureMapExtractor,
    get_content_image_transformer,
    get_style_image_transformer,
)

# "We normalized the network by scaling the weights such that the mean activation of each convolutional filter over images and positions is equal to one. Such re-scaling can be done for the VGG network without changing its output, because it contains only rectifying linear activation functions and no normalization or pooling over feature maps. We do not use any of the fully connected layers."


class ContentLoss(nn.Module):
    def __init__(self, model, layer):
        super().__init__()

        self.model = model
        self.layer = layer

        self.content_feat_map = FeatureMapExtractor(model).get_feature_map(image=content_image, layer=layer)

    def forward(self, gen_image):
        gen_feat_map = FeatureMapExtractor(self.model).get_feature_map(image=gen_image, layer=self.layer)
        # "$L_{content}(\vec{x}, \vec{c}, l) = \frac{1}{2}\sum_{i, j}\big(F^{l}_{x, ij} - F^{l}_{c, ij}\big)^{2}$"
        x = (1 / 2) * F.mse_loss(gen_feat_map, self.content_feat_map, reduction="sum")
        return x


def get_gram_matrix(feat_map):
    b, c, _, _ = feat_map.shape
    # "A layer with $N_{l}$ distinct filters has $N_{l}$ feature maps each of size $M_{l}$, where $M_{l}$ is the height times the width of the feature map. So the responses in a layer $l$ can be stored in a matrix $F^{l} \in \mathbb{R}^{N_{l} \times M_{l}}$ where $F^{l}_{ij}$ is the activation of the $i$ th filter at position $j$ in layer $l$."
    # "$G^{l}_{ij} = \sum_{k}F^{l}_{ik}F^{l}_{jk}$"
    x1 = feat_map.view((b, c, -1))
    x = torch.matmul(x1, torch.transpose(x1, dim0=1, dim1=2))
    return x


# "$w_{l}$"
def _get_contribution_of_layer(feat_map1, feat_map2):
    gram_mat1 = get_gram_matrix(feat_map1)
    gram_mat2 = get_gram_matrix(feat_map2)

    _, c, h, w = feat_map1.shape
    # "$E_{l} = \frac{1}{4N_{l}^{2}M_{l}^{2}} \sum_{i, j}\big(G^{l}_{x, ij} - G^{l}_{s, ij}\big)^{2}$"
    contrib = 1 / (4 * (c ** 2) * ((h * w) ** 2)) * F.mse_loss(gram_mat1, gram_mat2, reduction="sum")
    return contrib


class StyleLoss(nn.Module):
    def __init__(self, model, weights, layers):
        super().__init__()

        self.model = model
        self.weights = weights
        self.layers = layers

        self.style_feat_maps = [
            FeatureMapExtractor(model).get_feature_map(image=style_image, layer=layer)
            for layer in self.layers
        ]

    def forward(self, gen_image):
        # "$L_{style}(\vec{x}, \vec{s}) = \sum_{l = 0}^{L}w_{l}E_{l}$"
        x = 0
        for weight, layer, style_feat_map in zip(self.weights, self.layers, self.style_feat_maps):
            gen_feat_map = FeatureMapExtractor(self.model).get_feature_map(image=gen_image, layer=layer)
            contrib = _get_contribution_of_layer(feat_map1=gen_feat_map, feat_map2=style_feat_map)
            x += weight * contrib
        return x


class TotalLoss(nn.Module):
    def __init__(
        self,
        model,
        # "Along the processing hierarchy of the network, the input image is transformed into representations that are increasingly sensitive to the actual content of the image, but become relatively invariant to its precise appearance. Thus, higher layers in the network capture the high-level content in terms of objects and their arrangement in the input image but do not constrain the exact pixel values of the reconstruction very much. In contrast, reconstructions from the lower layers simply reproduce the exact pixel values of the original image. We therefore refer to the feature responses in higher layers of the network as the content representation."
        content_layer="features.40",
        # "$w_{l} = \frac{1}{5}$
        # in layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1'."
        style_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
        style_layers=("features.0", "features.7", "features.14", "features.27", "features.40"),
        lamb=1 * 1e-2, # "$Reciprocal of \frac{\alpha}{\beta}$" ($\lambda$)
    ):
        super().__init__()

        self.model = model
        self.content_layer = content_layer
        self.style_weights = style_weights
        self.style_layers = style_layers
        self.lamb = lamb

        self.content_loss = ContentLoss(model=model, layer=content_layer)
        self.style_loss = StyleLoss(model=model, weights=style_weights, layers=style_layers)
    
    def forward(self, gen_image):
        assert (
            gen_image.shape[0] == 1 and content_image.shape[0] == 1 and style_image.shape[0] == 1,
            "The batch size should be 1!"
        )

        x1 = self.content_loss(gen_image)
        x2 = self.style_loss(gen_image)
        x = self.lamb * x1 + x2
        return x


def prepare_images(args):
    content_img = load_image(args.content_image)
    style_img = load_image(args.style_image)
    gen_img = get_white_noise(content_img)

    content_transform = get_content_image_transformer()
    style_transform = get_style_image_transformer(content_img)

    content_image = content_transform(content_img).unsqueeze(0)
    style_image = style_transform(style_img).unsqueeze(0)
    gen_image = content_transform(gen_img).unsqueeze(0)
    return content_image, style_image, gen_image


if __name__ == "__main__":
    args = get_args()

    content_image, style_image, gen_image = prepare_images(args)

    cuda = torch.cuda.is_available()
    if cuda:
        content_image = content_image.cuda()
        style_image = style_image.cuda()
        gen_image = gen_image.cuda()
    # temp = tensor_to_array(gen_image)
    # show_image(temp)

    # "We used the feature space provided by a normalised version of the 16 convolutional and 5 pooling layers
    # of the 19-layer VGG network."
    # Max Pooling을 그대로 사용하겠습니다. ("For image synthesis we found that replacing the maximum pooling operation by average pooling yields slightly more appealing results, which is why the images shown were generated with average pooling.")
    model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
    model.eval()
    if cuda:
        model = model.cuda()

    feat_map_extractor = FeatureMapExtractor(model)

    gen_image.requires_grad_()
    # Adam을 사용하겠습니다. ("Here we use L-BFGS [32], which we found to work best for image synthesis.")
    optimizer = optim.Adam(params=[gen_image], lr=0.03)
    # optimizer = optim.LBFGS(params=[gen_image])

    criterion = TotalLoss(model=model, lamb=args.style_weight)

    n_epochs = 30_000
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        loss = criterion(gen_image)

        loss.backward()

        optimizer.step()
        if epoch % 200 == 0:
            print(f"""| Epoch: {epoch:5d} | Loss: {loss.item(): .2f} |""")

            gen_img = tensor_to_array(gen_image)
            save_image(
                img=gen_img,
                path=Path(args.save_dir)/f"""{Path(args.content_image).stem}_{Path(args.style_image).stem}_lambda{args.style_weight}_epoch{epoch}.jpg"""
            )
