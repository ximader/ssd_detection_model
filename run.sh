#!/bin/sh

python src/utils/download.py "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "./dataset/"
python src/utils/download.py "http://images.cocodataset.org/zips/val2017.zip" "./dataset/"
python src/utils/unzip.py "./dataset/annotations_trainval2017.zip" "./dataset/annotations_trainval2017"
python src/utils/unzip.py "./dataset/val2017.zip" "./dataset/val2017"
python train.py
python detect.py

read -p "Press enter to continue"
 