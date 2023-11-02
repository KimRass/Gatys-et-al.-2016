#!/bin/bash

data_dir=$1

for content_img_path in "$data_dir/content_images"/*; do
    for style_img_path in "$data_dir/style_images"/*; do
        python3 main.py\
            --content_img=$content_img_path\
            --style_img $style_img_path\
            --save_dir=$data_dir
    done
done