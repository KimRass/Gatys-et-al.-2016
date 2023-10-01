# Image-Style-Transfer
- Gatys et al. (2016) implementation from scratch in PyTorch
## Paper Reading
- [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
### Loss
- Content loss
    $$L_{content}(\vec{x}, \vec{c}, l) = \frac{1}{2}\sum_{i, j}\big(F^{l}_{x, ij} - F^{l}_{c, ij}\big)^{2}$$
    - Thus we can change the initially random image $\vec{x}$ until it generates the same response in a certain layer of the Convolutional Neural Network as the content image $\vec{c}$.
- Style loss
    $$G^{l}_{ij} = \sum_{k}F^{l}_{ik}F^{l}_{jk}$$
    $$E_{l} = \frac{1}{4N_{l}^{2}M_{l}^{2}} \sum_{i, j}\big(G^{l}_{x, ij} - G^{l}_{s, ij}\big)^{2}$$
    $$L_{style}(\vec{x}, \vec{s}) = \sum_{l = 0}^{L}w_{l}E_{l}$$
$$L(\vec{x}, \vec{c}, \vec{s}) = \lambda L_{content}(\vec{x}, \vec{c}) + L_{style}(\vec{x}, \vec{s})$$
### References
- [28] [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
## Research
### Model
- `from torchvision.models.vgg19_bn`을 사용하면 아래 이미지처럼 Style transfering이 제대로 일어나지 않습니다. `from torchvision.models.vgg19`을 사용하면 제대로 된 이미지가 생성되는 것을 확인할 수 있습니다.
    - <img src="https://github.com/KimRass/Gatys-et-al./assets/67457712/b6141441-89b7-4878-b058-c112c96af038" width="300">
- VGG19:
    - conv1_1: 0 ~ 1
    - conv1_2: 2 ~ 4
    - conv2_1: 5 ~ 6
    - conv2_2: 7 ~ 9
    - conv3_1: 10 ~ 11
    - conv3_2: 12 ~ 13
    - conv3_3: 14 ~ 15
    - conv3_4: 16 ~ 18
    - conv4_1: 19 ~ 20
    - conv4_2: 21 ~ 22
    - conv4_3: 23 ~ 24
    - conv4_4: 25 ~ 27
    - conv5_1: 28 ~ 29
    - conv5_2: 30 ~ 31
    - conv5_3: 32 ~ 33
    - conv5_4: 34 ~ 36

<!-- 
# Implementation from Scratch
## Samples (0.002)
- Content image
    - <img src="https://github.com/KimRass/image_style_transfer_from_scratch/assets/67457712/637040bd-b7c5-47e4-830d-deffa96454cc" width="600">
- Style image
    - <img src="https://github.com/KimRass/image_style_transfer_from_scratch/assets/67457712/dd398f2b-5215-4607-92af-bb6056f93041" width="600">
- Generated image
    - <img src="https://github.com/KimRass/image_style_transfer_from_scratch/assets/67457712/d28e7c7a-e02f-4907-aa2b-b11d0932b9d2" width="600"> -->
