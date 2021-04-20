#!/bin/sh
mkdir training-data
mkdir testing-data
mkdir validation-data
echo "downloading dataset and splitting it into training, testing and validation data"
wget https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset.zip
if ! command -v unzip &> /dev/null
then
  apt-get install unzip -y
fi
unzip DeepVideoDeblurring_Dataset.zip
cd DeepVideoDeblurring_Dataset/DeepVideoDeblurring_Dataset/quantitative_datasets/
ls | shuf -n 50 | xargs -i mv {} ../../../training-data
ls | shuf -n 7 | xargs -i mv {} ../../../testing-data
ls | shuf -n 14 | xargs -i mv {} ../../../validation-data
cd ../../../
python3 patchify_images.py
echo "done!"
