#!/usr/bin/env python
import sys
import os
import os.path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import glob
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error
input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

# Cell
def getmap( coco_json,corrected_result_json,annType="bbox",ap=1, iouThr=None, areaRng='all', maxDets=200 ):


    cocoGt=COCO(str(coco_json))
    cocoDt=cocoGt.loadRes(str(corrected_result_json))
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    img_ids = cocoGt.getImgIds()

    cocoEval.params.imgIds = img_ids
    cocoEval.evaluate()

    cocoEval.accumulate()
    cocoEval.summarize()
    p = cocoEval.params

    maxDets=cocoEval.params.maxDets[2]

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = cocoEval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,:,aind,mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = cocoEval.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,aind,mind]
    if len(s[s>-1])==0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s>-1])
    return mean_s


def get_matches(gts, predictions,matching="iou",overlapThresh =0.5):

    gts = np.array([np.array(bbox) for bbox in gts ])

    boxes = np.array([np.array(bbox) for bbox in predictions ])
    # initialize the list of picked indexes	
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]+boxes[:,0]
    y2 = boxes[:,3]+boxes[:,1]


    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)


    # keep looping while some indexes still remain in the indexes
    # list
    idxs = list(range(len(area)))
    for (x,y,h,w) in gts:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        xx = x+h
        yy = y+w

        aaa = h*w


        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x, x1[idxs])
        yy1 = np.maximum(y, y1[idxs])
        xx2 = np.minimum(xx, x2[idxs])
        yy2 = np.minimum(yy, y2[idxs])

        # compute the width and height of the bounding box
        ww = np.maximum(0, xx2 - xx1 + 1)
        hh = np.maximum(0, yy2 - yy1 + 1)

        # compute intersection over union (union is area 1 +area 2-intersection)
        if matching =="iou":
            overlap = (ww * hh) / (area[idxs]+aaa -(ww*hh))
        elif matching =="iom":
            overlap = (ww*hh) / np.minimum(aaa,area[idxs])
        true_matches = np.where(overlap > overlapThresh)

        if len(true_matches[0]) > 0:
            pick.append(idxs[true_matches[0][0]])

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, true_matches[0])


    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


# Cell

def get_det_metrics(reference_json,prediction_json,iou_thr=0.5,matching="iou"):
    boxes_true = {}
    boxes_pred = {}
    area = {}

    for img in reference_json["images"]:
        boxes_true[img["id"]] = []
        boxes_pred[img["id"]] = []


    for ann in reference_json["annotations"]:
        boxes_true[ann["image_id"]].append(ann["bbox"])





    for ann in prediction_json:
        if ann["score"] >0.5:
            boxes_pred[ann["image_id"]].append(ann["bbox"]+[ann["score"]])

    for img_id in sorted(boxes_pred):
        boxes = boxes_pred[img_id]

        #pick = non_max_suppression_iom(boxes)
        #boxes = np.array(boxes)[pick]
        boxes_pred[img_id] = [[x,y,h,w] for (x,y,h,w,_) in boxes]


    tp = 0
    fn = 0
    fp = 0
    for img_id in sorted(boxes_true):
        if len(boxes_pred[img_id]) >0:
            pick = get_matches(boxes_true[img_id], boxes_pred[img_id],matching,overlapThresh=iou_thr)

            tp += len(pick)




            fn += len(boxes_true[img_id]) - len(pick)
            fp += len(boxes_pred[img_id]) - len(pick)

        else:
            tp += 0
            fn += len(boxes_true[img_id])
            fp += 0


    return float(tp) / (float(tp)+float(fn)+float(fp))


# Cell

def kaggle_map(reference_json,prediction_json):

    with open(prediction_json) as f:
        prediction_data = json.load(f)
        
    with open(reference_json) as g:
        reference_data = json.load(g)



    mAP = np.mean([get_det_metrics(reference_data,prediction_data,iou_thr=i) for i in np.arange(0.5,0.76,0.05)])

    return mAP

def get_accuracy(reference_json,prediction_json,iou_thr=0.5):
    with open(prediction_json) as f:
        prediction_data = json.load(f)
        
    with open(reference_json) as g:
        reference_data = json.load(g)

    
    return get_det_metrics(reference_data,prediction_data,iou_thr=iou_thr)


if not os.path.isdir(submit_dir):
    print "%s doesn't exist" % submit_dir

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')
    




    original_json=[]
    for file in os.listdir(truth_dir):
        if file.endswith(".json"):
            original_json.append(os.path.join(truth_dir,file))

    original_json = sorted(original_json)

    prediction_json=[]

    for file in os.listdir(submit_dir):
        if file.endswith(".json"):
            prediction_json.append(os.path.join(submit_dir,file))
    prediction_json = sorted(prediction_json)

    if len(original_json) == 4:
        map_ = 0
        k_map = 0
        accuracy = 0

        for gt,pred in zip(original_json,prediction_json):

            map_ += getmap(gt,pred,"bbox",1, iouThr=0.5)
            k_map += kaggle_map(gt,pred)
            accuracy += get_accuracy(gt,pred)


        map_ = map_ / len(original_json)
        k_map = k_map / len(original_json)
        accuracy = accuracy / len(original_json)
    else:
        map_ = 0.0
        k_map = 0.0
        accuracy = 0.0

    if os.path.isfile(os.path.join(submit_dir,"count.csv")):
        original_count=pd.read_csv(os.path.join(truth_dir,"count.csv"))

        pred_count=pd.read_csv(os.path.join(submit_dir,"count.csv"))

        final_frame =pd.merge(original_count,pred_count,on="image_name")

        sessions = list(set(original_count["session"]))

        rmse = 0
        for sess in sessions:
            sub_df = final_frame.query("session_x == '%s'" % sess)
            true = final_frame["count_x"].values
            pred = final_frame["count_y"].values

            
            rmse += np.sqrt(mean_squared_error(true, pred))
        rmse = round(rmse / len(sessions),4)
    else:
        rmse = 100


    output_file.write("mAP@0.5:%s\nKaggle Accuracy:%s\nAccuracy:%s\nRMSE:%s" % (map_,k_map,accuracy,rmse) )