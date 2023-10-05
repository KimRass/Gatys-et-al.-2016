# Gatys-et-al.
- Gatys et al. (2016) implementation from scratch in PyTorch
## Paper Reading
- [Image Style Transfer Using Convolutional Neural Networks](https://github.com/KimRass/Gatys-et-al./blob/main/image_style_transfer_using_convolutional_neural_networks.pdf)
### References
- [28] [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
## Research
### Model
- `from torchvision.models.vgg19_bn`을 사용하면 아래 이미지처럼 style transferring이 제대로 일어나지 않습니다. `from torchvision.models.vgg19`을 사용하면 제대로 된 이미지가 생성되는 것을 확인할 수 있습니다.
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
### Input Image Size
- VGG19가 224 × 224의 input size로 학습되었기 때문인지, 이와 비슷한 크기로 conetent image와 style image로 resize하지 않으면 즉 224 × 224보다 훨씬 큰 input size로 style transferring을 시도하면 제대 된 이미지가 생성되지 않음을 확인했습니다.
