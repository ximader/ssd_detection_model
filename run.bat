call python src/utils/download.py "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "./dataset/" 
call python src/utils/download.py "http://images.cocodataset.org/zips/val2017.zip" "./dataset/" 

call python src/utils/unzip.py "./dataset/annotations_trainval2017.zip" "./dataset/annotations_trainval2017" 
call python src/utils/unzip.py "./dataset/val2017.zip" "./dataset/val2017" 


call conda activate cv

call python train.py

call python detect.py

pause