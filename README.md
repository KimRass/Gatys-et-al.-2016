- [Image Style Transfer Using Convolutional Neural Networks](https://github.com/KimRass/Gatys-et-al.-2016/blob/main/papers/image_style_transfer_using_convolutional_neural_networks.pdf)

# 1. How to Generate Images
- Single image
    ```bash
    # e.g.,
    python3 main.py\
        --content_img="examples/content_images/content_image1.jpg"\
        --style_img="examples/style_images/style_image1.jpg"\
        --save_dir="examples/"\
        --n_epochs=400\ # Optional
        --alpha=1\ # Optional
        --beta=100000000\ # Optional
        --from_noise # 완전한 gaussian noise로부터 이미지를 생성하기 위해서는 상대적으로 매우 큰 `n_epochs` 값이 필요합니다.
    ```
- Multiple images
    ```bash
    # e.g.,
    bash generate_images.sh --data_dir="path/to/images/dir"
    ```

# 2. Samples
- <img src="https://github.com/KimRass/PGGAN/assets/105417680/254ee5cc-0220-4e74-a0ee-635b53157703" width="800">
- <img src="https://github.com/KimRass/PGGAN/assets/105417680/690e02d2-d762-49ac-b2ac-79df6d1e7bf1" width="800">
- <img src="https://github.com/KimRass/PGGAN/assets/105417680/f6f7fc96-cca9-495a-a065-9c10d243400b" width="800">
- <img src="https://github.com/KimRass/PGGAN/assets/105417680/06c8d9c9-9e20-4cae-b40a-70167aba9b84" width="800">
- <img src="https://github.com/KimRass/PGGAN/assets/105417680/4bd17692-44ec-4f47-a040-dc671d2147ff" width="800">

# 3. Implementation Details
## 1) Architecture
- 논문에 다음과 같은 말이 있습니다.
    - "We used the feature space provided by a normalised version of the 16 convolutional and 5 pooling layers of the 19-layer VGG network."
- Batch normalization을 사용한 버전을 얘기하는 건가 싶어서 `torchvision.models.vgg19_bn`을 사용하면 아래 이미지처럼 style transferring이 제대로 일어나지 않습니다. `torchvision.models.vgg19`을 사용하면 제대로 된 이미지가 생성됩니다.
    - <img src="https://github.com/KimRass/Gatys-et-al./assets/67457712/b6141441-89b7-4878-b058-c112c96af038" width="300">
## 2) Input Image Size
- VGG19가 224 × 224의 input size로 학습되었기 때문인지, 이와 비슷한 크기로 conetent image와 style image로 resize하지 않으면 즉 224 × 224보다 훨씬 큰 input size로 style transferring을 시도하면 제대 된 이미지가 생성되지 않음을 확인했습니다.
## 3) Normalization
- 논문에 다음과 같은 말이 있습니다.
    - "We normalized the network by scaling the weights such that the mean activation of each convolutional filter over images and positions is equal to one. Such re-scaling can be done for the VGG network without changing its output, because it contains only rectifying linear activation functions and no normalization or pooling over feature maps. We do not use any of the fully connected layers."
- 그런데 이 말이 무엇을 의미하는 지 개인적으로 이해하지 못 했고, 또한 다른 분들이 구현한 코드를 보아도 이와 관련된 부분은 없는 것 같아 저도 별도로 구현하지 않았습니다.

# 4. VGG19 Architecture
```bash
# from torchvision.models import vgg19

VGG(
    (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv1_1
        (1): ReLU(inplace=True) # conv1_1
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv1_2
        (3): ReLU(inplace=True) # conv1_2
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) # conv1_2
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv2_1
        (6): ReLU(inplace=True) # conv2_1
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv2_2
        (8): ReLU(inplace=True) # conv2_2
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) # conv2_2
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv3_1
        (11): ReLU(inplace=True) # conv3_1
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv3_2
        (13): ReLU(inplace=True) # conv3_2
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv3_3
        (15): ReLU(inplace=True) # conv3_3
        (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv3_4
        (17): ReLU(inplace=True) # conv3_4
        (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) # conv3_4
        (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv4_1
        (20): ReLU(inplace=True) # conv4_1
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv4_2
        (22): ReLU(inplace=True) # conv4_2
        (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv4_3
        (24): ReLU(inplace=True) # conv4_3
        (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv4_4
        (26): ReLU(inplace=True) # conv4_4
        (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) # conv4_4
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv5_1
        (29): ReLU(inplace=True) # conv5_1
        (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv5_2
        (31): ReLU(inplace=True) # conv5_2
        (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv5_3
        (33): ReLU(inplace=True) # conv5_3
        (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # conv5_4
        (35): ReLU(inplace=True) # conv5_4
        (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) # conv5_4
    )
)
```