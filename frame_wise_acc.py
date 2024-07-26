import pandas as pd
import numpy as np
import argparse

def compute_frame_wise(ground_truth, prediction):
    scores = np.zeros(len(ground_truth))
    for i in range(len(ground_truth)):
        if ground_truth[i] == prediction[i]:
            scores[i] = 1.0
    return np.mean(scores)

parser = argparse.ArgumentParser("Calculate Frame-wise Accuracy")
parser.add_argument('--pred', type=str, default=".data/cls.csv")
parser.add_argument('--gt', type=str, default='./data/ground_truth')

gt = pd.read_csv(str(parser.parse_args().gt))
prediction = pd.read_csv(str(parser.parse_args().pred))
M, N = len(gt), len(prediction)
gt_by_label = gt.groupby("video_id")
pred_by_label = prediction.groupby("video_id")
scores = []
id_video = gt['video_id'].unique().tolist()
for label in id_video:
        ground_truth_class = gt_by_label.get_group(label).reset_index(drop=True)
        prediction_class = pred_by_label.get_group(label).reset_index(drop=True) 
        gt_list = np.zeros((int(prediction_class['end'].iloc[-1])))
        pred_list = np.zeros_like(gt_list)
        for idx, this_gt in ground_truth_class.iterrows():
            start = this_gt['start']
            end = this_gt['end']
            gt_list[start:end+1] = int(this_gt['label'])
        for idx, this_pred in prediction_class.iterrows():
            start = this_pred['start']
            end = this_pred['end']
            pred_list[start:end+1] = int(this_pred['label'])
        scores.append(compute_frame_wise(gt_list, pred_list))
print(sum(scores)/(len(scores)))