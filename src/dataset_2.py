from torch.utils.data import Dataset
import torch
from pycocotools.coco import COCO
import cv2
import os
import numpy as np
import albumentations as A

class FilteredCocoDataset(Dataset):
    def __init__(self, category_name,  annot_json, images_folder, im_height, im_width, transforms=None):
        """
        Initialize Filtered COCO dataset

        Args:
            category (str): category name, like 'person'    
            annot_json (str): path to json
            images_folder (str): path to images folder
            im_height (int): output image height
            im_width (int): output image width
            transforms: optional albumentation transforms 
        
        """
        self.annot_json = annot_json
        self.images_folder = images_folder
        self.transforms = transforms
        self.category_names = [category_name]
        self.im_height = im_height
        self.im_width = im_width

        self.coco = COCO(self.annot_json)                                       # get COCO annotation browser                  
        self.category_ids = self.coco.getCatIds(catNms=self.category_names)     # get list of filtered category Ids
        self.image_ids = self.coco.getImgIds(catIds=self.category_ids)          # filter image Ids with desired categories



    def __len__(self):
        return len(self.image_ids)
    

        
    def __getitem__(self, idx):

        # load image data
        """[{
            'license': 1, 
            'file_name': '000000458755.jpg', 
            'coco_url': 'http://images.cocodataset.org/val2017/000000458755.jpg', 
            'height': 480, 
            'width': 640, 
            'date_captured': '2013-11-16 23:06:51', 
            'flickr_url': 'http://farm6.staticflickr.com/5119/5878453277_eea657a01d_z.jpg', 
            'id': 458755
            }]
        """
        img_id = self.image_ids[idx]                            # image unique Id
        img_data = self.coco.loadImgs(img_id)[0]                # image metadata from dataset
        file_name = img_data['file_name']
        img_path = os.path.abspath(self.images_folder+file_name).replace('\\', '/')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)          # load image from path
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0       
        image = np.float32(image)

        # load image annotations
        """{
            'segmentation': [[10.32, 252.9, 39.23, 245.68, 76.39, 247.74, 107.35, 250.84, 140.39, 266.32, 163.1, 272.52, 179.61, 281.81, 192.0, 293.16, 214.71, 310.71, 231.23, 318.97, 250.84, 327.23, 261.16, 329.29, 293.16, 333.42, 309.68, 324.13, 317.94, 308.65, 322.06, 293.16, 322.06, 276.65, 321.03, 261.16, 312.77, 239.48, 299.35, 214.71, 279.74, 178.58, 264.26, 162.06, 245.68, 151.74, 210.58, 134.19, 165.16, 110.45, 115.61, 87.74, 98.06, 73.29, 51.61, 48.52, 16.52, 27.87, 1.03, 28.9, 4.13, 86.71, 3.1, 138.32, 4.13, 165.16, 4.13, 192.0, 4.13, 217.81, 4.13, 245.68, 6.19, 256.0]], 
            'area': 54043.235450000015, 
            'iscrowd': 0, 
            'image_id': 458755, 
            'bbox': [1.03, 27.87, 321.03, 305.55], 
            'category_id': 20, 
            'id': 61049}
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id)            # get all annotation Ids for specific image
        annotations = []
        for ann in [self.coco.anns[x] for x in ann_ids]:
            if ann['category_id'] in self.category_ids:
                annotations.append(ann)


        # get bounding boxes
        bboxes = [x['bbox'] for x in annotations]

        # get categories
        labels = [x['category_id'] for x in annotations]

        if self.transforms is not None: 
            transformed = self.transforms(image=image, bboxes=bboxes, class_labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        return (
            torch.from_numpy(image).permute(2, 0, 1).float(), 
            [tuple(self.convert_bbox(box)) for box in bboxes], 
            labels)



    def convert_bbox(self, bbox):
        """Convert bbox size from absolute x,y,w,h to relative x_min, y_min, x_max, y_max"""
        x_min,y_min,w,h = bbox
        x_max, y_max = x_min+w, y_min+h
        return torch.FloatTensor((x_min/self.im_width, 
                                  y_min/self.im_height, 
                                  x_max/self.im_width, 
                                  y_max/self.im_height))
    
    

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Annotations JSON Path: {}\n'.format(self.annot_json)
        fmt_str += '    Images filder location: {}\n'.format(self.images_folder)
        return fmt_str

        

if __name__=="__main__":

    images_folder = './dataset/val2017/val2017/'
    annot_json = './dataset/annotations_trainval2017/annotations/instances_val2017.json'
    CATEGORY = 'person'
    IMAGE_WIDTH = 300
    IMAGE_HEIGHT = 300

    transform = A.Compose([
        A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter() 
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))


    full_dataset = FilteredCocoDataset(CATEGORY, annot_json, images_folder, IMAGE_WIDTH, IMAGE_HEIGHT, transform)
    print(full_dataset.category_names, full_dataset.category_ids)
    print(full_dataset)

    img, bboxes, labels = full_dataset[1]
    print(f'{img.shape = }')
    print(f'{bboxes = }')
    print(f'{labels = }')