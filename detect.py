import numpy as np
import cv2
import torch
from src.config import *
from src.draw_bboxes import draw_bboxes
from argparse import ArgumentParser
import sys
from src.ssdmodel import  SimpleSSD
import glob
import os

def detect(model, image, min_score=0.4, max_overlap=0.5, top_k=100):
    """ 
    Detect objects on single image 

    Args:
        model: Simple SSD mmodel
        orig_image: input image in numpy RGB format
        min_score: minimum confidence score to detect object
        max_overlap: max overlap to non-maximum supression
        top_k: maximum amount of bboxes

    Return:
        image: annotated image
        
    """

    # prepare input image
    if np.max(image)>1:
        image = image/255.
    
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    tensor = torch.tensor(image).permute(2,0,1).float().unsqueeze(0).to(model.device)

    # make predictions
    predicted_locs, predicted_scores = model(tensor)

    # detect bboxes ordered by confidence score
    det_boxes, det_labels, det_scores = model.detect_objects(
        predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap, top_k=top_k)

    # get detections for single image
    det_boxes = det_boxes[0].to('cpu').detach().numpy()
    det_labels = det_labels[0].to('cpu').detach().numpy()
    det_scores = det_scores[0].to('cpu').detach().numpy()

    # filter only objects
    for i in range(len(det_boxes)):
        if det_labels[i] != 0:
            image = draw_bboxes(image, [det_boxes[i]], [det_labels[i]], [det_scores[i]])

    return image




def parse_args():
    parser = ArgumentParser(description="Detect objects")
    parser.add_argument('-weights', type=str, default='models/model.pth', help='Location of model.pth')
    parser.add_argument("-folder", type=str, default='test_images', help="Folder to read images")
    parser.add_argument("-output", type=str, default='output', help="Folder to save annotated images")
    return parser.parse_args()



if __name__=="__main__":
    # parse arguments
    try: 
        args = parse_args()
    except Exception as e:
        print('Arguments parsing error.\n' + str(e), file=sys.stderr)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = SimpleSSD(2, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(args.weights))


    # get files from folder 
    types = ('*.jpg', '*.png', '*.bmp', '*.jpeg')  
    files = []
    for fmask in types:
        files.extend(glob.glob( os.path.join(args.folder, fmask)))
    

    for file in files:
        file = file.replace('\\', '/')
        origin_img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        img = detect(model, img, min_score=0.3)

        fname = 'annot_'+file.split('/')[-1]
        cv2.imwrite(os.path.join(args.output,fname), np.uint8(img[:,:,::-1]*255))
        print(f'{os.path.join(args.output,fname)} saved.')
 