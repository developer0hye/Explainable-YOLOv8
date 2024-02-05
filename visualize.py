# author: Yonghye Kwon
# email: developer.0hye@gmail.com
import argparse
import pathlib

import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils.tal import make_anchors

class DetailedResults(Results):
    def __init__(self, orig_img, path, names, input, preprocessed_preds, raw_preds, boxes=None, masks=None, probs=None, keypoints=None, obb=None) -> None:
        super().__init__(orig_img, path, names, boxes, masks, probs, keypoints, obb)
        self.input = input
        self.preprocessed_preds = preprocessed_preds
        self.raw_preds = raw_preds

def postprocess(self, preds, img, orig_imgs):
    """Post-processes predictions and returns a list of Results objects."""
    preprocessed_preds, raw_preds = preds
    
    preds = ops.non_max_suppression(
        preprocessed_preds.clone(),
        self.args.conf,
        self.args.iou,
        agnostic=self.args.agnostic_nms,
        max_det=self.args.max_det,
        classes=self.args.classes,
    )

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        preprocessed_preds = preprocessed_preds[0]
        preprocessed_preds = preprocessed_preds.transpose(0, 1)
        preprocessed_preds[:, :4] = ops.scale_boxes(img.shape[2:], preprocessed_preds[:, :4], orig_img.shape, xywh=True)
        
        img_path = self.batch[0][i]
        results.append(DetailedResults(orig_img, path=img_path, names=self.model.names, input=img, preprocessed_preds=preprocessed_preds, raw_preds=raw_preds, boxes=pred))
    return results

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='yolov8m.pt', help='model.pt path')
argparser.add_argument('--source', type=str, default='./zidane.jpg', help='source')  # Refer to https://docs.ultralytics.com/modes/predict/#inference-sources
argparser.add_argument('--verbose', action='store_true', help='verbose')
argparser.add_argument('--class_id', type=int, default=0, help='class id to visualize')
argparser.add_argument('--thr_visualize_conf', type=float, default=0.5, help='threshold for visualizing anchor xy')
argparser.add_argument('--thr_visualize_ltrb', type=float, default=0.5, help='threshold for visualizing ltrb')
args = argparser.parse_args()

model = YOLO(args.model)  # initialize
names = model.model.names
num_classes = len(names)

model(args.source, verbose=args.verbose)  # dummy inference to setup predictor
model.predictor.postprocess = postprocess.__get__(model.predictor)  # monkey patch
results = model(args.source, verbose=args.verbose)  # inference

for result in results: 
    anchors, strides = make_anchors(result.raw_preds, strides=[8, 16, 32])
    xy = anchors * strides # the poosition of the anchor
    wh = torch.zeros_like(xy)
    
    xywh = torch.cat((xy, wh), dim=-1)
    xywh = ops.scale_boxes(result.input.shape[2:], xywh, result.orig_img.shape, xywh=True)
    
    xywh = xywh.cpu().numpy() 
    
    boxes_xywh = result.preprocessed_preds[:, :4]
    boxes_xywh = boxes_xywh.cpu().numpy()

    boxes_xyxy = ops.xywh2xyxy(boxes_xywh)
    boxes_ltrb = np.zeros_like(boxes_xyxy)
    
    boxes_ltrb[:, 0] = xywh[:, 0] - boxes_xyxy[:, 0]
    boxes_ltrb[:, 1] = xywh[:, 1] - boxes_xyxy[:, 1]
    boxes_ltrb[:, 2] = boxes_xyxy[:, 2] - xywh[:, 0]
    boxes_ltrb[:, 3] = boxes_xyxy[:, 3] - xywh[:, 1]
    
    preds_cat_conf_score = result.preprocessed_preds[:, 4:]
    preds_cat_conf_score = preds_cat_conf_score.cpu().numpy()
    
    # draw line on image
    for i in range(xywh.shape[0]):
        if preds_cat_conf_score[i][args.class_id] < args.thr_visualize_ltrb:
            continue
        
        x, y, _, _ = xywh[i]
         
        l, t, r, b = boxes_ltrb[i]

        cv2.rectangle(result.orig_img, (int(x - l), int(y - t)), (int(x + r), int(y + b)), (0, int(255 * preds_cat_conf_score[i][args.class_id]), 0), 2)
       
        # draw ltrb using arrow
        cv2.arrowedLine(result.orig_img, (int(x), int(y)), (int(x - l), int(y)), (0, 0, int(255 * preds_cat_conf_score[i][args.class_id])), 2)
        cv2.arrowedLine(result.orig_img, (int(x), int(y)), (int(x), int(y - t)), (int(255 * preds_cat_conf_score[i][args.class_id]), 0, int(255 * preds_cat_conf_score[i][args.class_id])), 2)
        cv2.arrowedLine(result.orig_img, (int(x), int(y)), (int(x + r), int(y)), (0, int(255 * preds_cat_conf_score[i][args.class_id]), int(255 * preds_cat_conf_score[i][args.class_id])), 2)
        cv2.arrowedLine(result.orig_img, (int(x), int(y)), (int(x), int(y + b)), (int(255 * preds_cat_conf_score[i][args.class_id]), 0, 0), 2)

    # draw anchor xy on image
    for i in range(xywh.shape[0]):
        if preds_cat_conf_score[i][args.class_id] < args.thr_visualize_conf:
            continue
        
        x, y, _, _ = xywh[i]

        # draw anchor xy
        cv2.circle(result.orig_img, (int(x), int(y)), 4, (0, int(255 * preds_cat_conf_score[i][args.class_id]), 0), -1)
    cv2.imwrite(f'result_{pathlib.Path(result.path).stem}_classid-{args.class_id:03d}_classname-{names[args.class_id]}.jpg', result.orig_img)


