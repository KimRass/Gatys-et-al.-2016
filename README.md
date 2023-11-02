# Gatys et al. (2016) implementation from scratch in PyTorch
- [Image Style Transfer Using Convolutional Neural Networks](https://github.com/KimRass/Gatys-et-al./blob/main/image_style_transfer_using_convolutional_neural_networks.pdf)
## How to Generate Images
```bash
# To generate single image, e.g., run
python3 main.py\
    --content_img="path/to/content/image.jpg"\
    --style_img="path/to/style/image.jpg"\
    --save_dir="dir/to/save/generated/images"\
    --n_epochs=400 # Optional
    --alpha=1\ # Optional
    --beta=100000000 # Optional
    --from_noise # 완전한 gaussian noise로부터 이미지를 생성하기 위해서는 상대적으로 매우 큰 `n_epochs` 값이 필요합니다.

# To generate multiple images, e.g., run
bash generate_images.sh --data_dir="path/to/images/dir"
```
## Sample Images
- <img src="https://github.com/KimRass/PGGAN/assets/105417680/254ee5cc-0220-4e74-a0ee-635b53157703" width="800">
- <img src="https://github.com/KimRass/PGGAN/assets/105417680/690e02d2-d762-49ac-b2ac-79df6d1e7bf1" width="800">
- <img src="https://github.com/KimRass/PGGAN/assets/105417680/f6f7fc96-cca9-495a-a065-9c10d243400b" width="800">
- <img src="https://github.com/KimRass/PGGAN/assets/105417680/06c8d9c9-9e20-4cae-b40a-70167aba9b84" width="800">
- <img src="https://github.com/KimRass/PGGAN/assets/105417680/4bd17692-44ec-4f47-a040-dc671d2147ff" width="800">
## VGG19:
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
## Implementation Details
### Model
- `from torchvision.models.vgg19_bn`을 사용하면 아래 이미지처럼 style transferring이 제대로 일어나지 않습니다. `from torchvision.models.vgg19`을 사용하면 제대로 된 이미지가 생성되는 것을 확인할 수 있습니다.
    - <img src="https://github.com/KimRass/Gatys-et-al./assets/67457712/b6141441-89b7-4878-b058-c112c96af038" width="300">
### Input Image Size
- VGG19가 224 × 224의 input size로 학습되었기 때문인지, 이와 비슷한 크기로 conetent image와 style image로 resize하지 않으면 즉 224 × 224보다 훨씬 큰 input size로 style transferring을 시도하면 제대 된 이미지가 생성되지 않음을 확인했습니다.
### Normalization
- 논문에 다음과 같은 말이 있습니다.
    - "We normalized the network by scaling the weights such that the mean activation of each convolutional filter over images and positions is equal to one. Such re-scaling can be done for the VGG network without changing its output, because it contains only rectifying linear activation functions and no normalization or pooling over feature maps. We do not use any of the fully connected layers."
- 그런데 이 말이 무엇을 의미하는 지 개인적으로 이해하지 못 했고, 또한 다른 분들이 구현한 코드를 보아도 이와 관련된 부분은 없는 것 같아 저도 별도로 구현하지 않았습니다.
