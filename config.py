# "The images were synthesised by matching the content representation on layer 'conv4_2' and the style
# representation on layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1'."
CONTENT_LAYER_NUMS = (21,)
STYLE_LAYERS_NUMS = (0, 5, 10, 19, 28)
LAYER_NUMS = set(CONTENT_LAYER_NUMS) | set(STYLE_LAYERS_NUMS)
# $w_{l} = 1 / 5$ in those layers, $w_{l} = 0$ in all other layers."
STYLE_WEIGHTS = (0.2, 0.2, 0.2, 0.2, 0.2)
assert len(STYLE_LAYERS_NUMS) == len(STYLE_WEIGHTS),\
    "`len(STYLE_LAYERS_NUMS)` should be equal to `len(STYLE_LAYERS_NUMS)`"
assert sum(STYLE_WEIGHTS) == 1, "`sum(STYLE_LAYERS_NUMS)` should be equal to 1"
MEAN = (0.485, 0.456, 0.406)
STD = [0.229, 0.224, 0.225]
LR = 1
N_EPOCHS = 150
FROM_CONTENT_IMAGE = True
