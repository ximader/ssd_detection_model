import cv2

def draw_bboxes(image, bboxes, labels = None, scores = None, color = (1., 0., 1.), thickness=2, lineType=cv2.LINE_8):
    image_annotated = image.copy()
    h, w = image.shape[:2]
    
    scale = w // 300
    
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
        x_min, y_min, x_max, y_max = int(x_min*w), int(y_min*h), int(x_max*w), int(y_max*h)
        image_annotated = cv2.rectangle(image_annotated, 
                                        (x_min, y_min), 
                                        (x_max, y_max), 
                                        color, 
                                        thickness=thickness*scale, 
                                        lineType=lineType)
        
        text = ''
        if labels is not None:
            text += f'{labels[i]:.0f} '

        if scores is not None:
            text += f'{scores[i]:.2f} '     

        if text != '':

            image_annotated = cv2.rectangle(image_annotated, 
                                            (x_min, y_min), (x_min+55*scale, y_min-15*scale), (1.,1.,1.), 
                                            thickness=-1, lineType=cv2.LINE_8)
            image_annotated = cv2.putText(image_annotated, text, 
                                      (x_min, y_min-2), cv2.FONT_HERSHEY_COMPLEX, 0.5*scale, (0.,0.,0.), 1*scale, cv2.LINE_AA);

    return image_annotated