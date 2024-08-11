import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from src.config import *
import albumentations as A
from src.dataset_2 import FilteredCocoDataset
from src.ssdmodel import SimpleSSD
from src.multibox_loss import MultiBoxLoss
from tqdm import tqdm
import os

# https://github.com/amdegroot/ssd.pytorch/blob/master/data/__init__.py#L9
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Args:
        batch: (tuple) A tuple of tensor images and lists of bboxes and labels

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) bboxes for a given image are stacked on 0 dim
            3) (list of tensors) labels for a given image are stacked on 0 dim
    """
    boxes = []
    labels = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        boxes.append(torch.FloatTensor(sample[1]))
        labels.append(torch.FloatTensor(sample[2]))
    return torch.stack(imgs, 0), boxes, labels



def train():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{DEVICE = }')

    # fix randomizers
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    # ============================================================================
    # get dataset
    transform = A.Compose([
        A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    full_dataset = FilteredCocoDataset(CATEGORY, ANNOT_JSON, IMAGES_FOLDER, IMAGE_WIDTH, IMAGE_HEIGHT, transform)
    print(full_dataset)


    # ============================================================================
    # get dataloaders
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # train
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=detection_collate)

    # test
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=detection_collate)
    test_dataset.transforms = None

    print('train_dataset images:', len(train_dataset))
    print('test_dataset images:', len(test_dataset))


    # ============================================================================
    # train
    model = SimpleSSD(2, DEVICE).to(DEVICE)
    loss_fn = MultiBoxLoss(model.prior_boxes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)


    print('','-'*25, f'\nStart training {EPOCHS} epochs\n','-'*25 )
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = []
        
        for step, (img, boxes, labels) in enumerate(pbar := tqdm(train_loader)):

            img = img.to(DEVICE)
            boxes = [box.to(DEVICE) for box in boxes]  
            labels = [label.to(DEVICE) for label in labels]  

            pred_loc, pred_sco = model(img)
            try:
                loss = loss_fn(pred_loc, pred_sco, boxes, labels)
                
                # Back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                train_loss.append(loss.item())

                descr = f"Train loss: {np.mean(train_loss):.4f}"
                pbar.set_description(descr)
            except:
                pass
        
        model.eval()
        valid_loss = []
        for step, (img, boxes, labels) in enumerate(tqdm(test_loader)):
            img = img.to(DEVICE)
            boxes = [box.to(DEVICE) for box in boxes]
            labels = [label.to(DEVICE) for label in labels]
            pred_loc, pred_sco = model(img)
            try:
                loss = loss_fn(pred_loc, pred_sco, boxes, labels)
                valid_loss.append(loss.item())
            except:
                pass
            
        print('epoch:', epoch, '/', EPOCHS,
                '\ttrain loss:', '{:.4f}'.format(np.mean(train_loss)),
                '\tvalid loss:', '{:.4f}'.format(np.mean(valid_loss)))
    
    # ============================================================================
    # save trained model
    torch.save(model.state_dict(), os.path.join('models', 'model.pth'))
    print('Trained model saved to:', os.path.join('models', 'model.pth'))



if __name__=='__main__':
    train()
