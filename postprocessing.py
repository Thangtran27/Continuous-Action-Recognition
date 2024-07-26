#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import pickle
import torch
import tqdm

import os 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import matplotlib.pyplot as plt
from tools import util_eval


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                ,[256,0,192],[250,100,150],[128,128,256]]

cmap = np.asarray(label_colours) / 255.0
ignore_file = 'data/error_arm.csv'
def key_sort(file):
    # basename = os.path.basename(file)
    first = str(file).split("_")[0][1:]
    last = str(file).split(".")[0][-1]
    number = int(first + last)
    return number
def read_ignore_file(file):
    data = pd.read_csv(file, header=None)
    list_start = [round(i/30) for i in data.iloc[:, 2].values.tolist()]
    list_subject = data.iloc[:, 0].values.tolist()
    subject = list(map(key_sort, list_subject))
    return subject, list_start # video_id

def process_overlap(data, name_vid, ignore_list_subject, ignore_list_start):
    data = data.sort_values(by=["start"]).reset_index(drop=True)
    for j in range(len(data)-2):
        if data.loc[j+1, "start"] - data.loc[j, "end"] < 0:
            data.loc[j, "end"] = 0
            data.loc[j, "start"] = 0
            # print(data.loc[j+1])
    duplicate_labels = data['label'][data['label'].duplicated()].unique()
    for i in duplicate_labels:
        for j in range(len(data)-1):
            label_a = int(data.loc[j, "label"])
            if label_a == i and j>=1:
                pre_end = int(data.loc[j-1, 'end'])
                former_start = int(data.loc[j, "start"])
                if pre_end == former_start:
                    data.loc[j, "end"] = 0
                    data.loc[j, "start"] = 0
    if name_vid in ignore_list_subject: # bỏ trong gesture bị lỗi
        start = ignore_list_start[ignore_list_subject.index(name_vid)]
        for j in range(len(data)-1):
            if abs(start - data.loc[j, 'start'])<=2: # int(data.loc[j, "label"]) in [11, 13] and 
                data.loc[j, "end"] = 0
                data.loc[j, "start"] = 0

    data = data[data["end"]!=0]            
    return data

def merge_and_remove(data, merge_threshold=16):
    df_total = pd.DataFrame([[0, 0, 0, 0]], columns=["video_id", "label", "start", "end"])
    data = data.reset_index(drop=True)
    data = data.sort_values(by=["video_id", "label"])
    ignore_list_subject, ignore_list_start = read_ignore_file(ignore_file)
    for i in data["video_id"].unique():
        data_video = data[data["video_id"]==i]
        list_label = data_video["label"].unique()
        vid_all = pd.DataFrame([[0, 0, 0, 0]], columns=["video_id", "label", "start", "end"])
        for label in list_label:

            data_video_label = data_video[data_video["label"]== label]
            data_video_label = data_video_label.reset_index()
            data_video_label = data_video_label.sort_values(by=["start"])

            for j in range(len(data_video_label)-1):

                if data_video_label.loc[j+1, "start"] - data_video_label.loc[j, "end"] <= merge_threshold:
                    data_video_label.loc[j+1, "start"] = data_video_label.loc[j, "start"]
                    data_video_label.loc[j, "end"] = 0
                    data_video_label.loc[j, "start"] = 0

            vid_all = vid_all.append(data_video_label)
        vid_all = vid_all[vid_all["end"]!=0]
        print("vid", i)
        vid_all = process_overlap(vid_all, i, ignore_list_subject, ignore_list_start)
        df_total = df_total.append(vid_all)
    df_total = df_total[df_total["end"]!=0]

    df_total = df_total.drop(columns=['index'])
    df_total = df_total.sort_values(by=["video_id", "start"])
    return df_total


def general_submission(data):
    # data = pd.read_csv(filename, sep=" ", header=None)
    data_filtered = data[data["label"] != 0]
    data_filtered["start"] = data["start"].map(lambda x: int(float(x)))
    data_filtered["end"] = data["end"].map(lambda x: int(float(x)))
    data_filtered = data_filtered.sort_values(by=["video_id","label"])
    results = merge_and_remove(data_filtered, merge_threshold=2)
    return results


def topk_by_partition(input, k, axis=None, ascending=False):
    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis) # k non-sorted indices
    input = np.take_along_axis(input, ind, axis=axis) # k non-sorted values

    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind_part, axis=axis) 
    return ind, val


def get_classification(sequence_class_prob):
    classify=[[x,y] for x,y in zip(np.argmax(sequence_class_prob, axis=1),np.max(sequence_class_prob, axis=1))]
    labels_index = np.argmax(sequence_class_prob, axis=1) #returns list of position of max value in each list.
    probs= np.max(sequence_class_prob, axis=1)  # return list of max value in each  list.
    return labels_index, probs

def activity_localization(prob_sq, action_threshold):
    action_idx, action_probs = get_classification(prob_sq)
    threshold = np.mean(action_probs)
    action_tag = np.zeros(action_idx.shape)
    # action_tag[action_probs >= threshold] = 1
    action_tag[action_probs >= action_threshold] = 1
    # print('action_tag', action_tag)
    activities_idx = []
    startings = []
    endings = []

    for i in range(len(action_tag)):
        if action_tag[i] ==1:
            activities_idx.append(action_idx[i])
            start = i
            end = i+1
            startings.append(start)
            endings.append(end)
    return activities_idx, startings, endings

def smoothing(x, k=3):
    ''' Applies a mean filter to an input sequence. The k value specifies the window
    size. window size = 2*k
    '''
    l = len(x)
    s = np.arange(-k, l - k)
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1
    y = np.zeros(x.shape)
    for i in range(l):
        y[i] = np.mean(x[s[i]:e[i]], axis=0)
    return y

def gauss_smoothing(x, k=3):
    ''' Applies a mean filter to an input sequence. The k value specifies the window
    size. window size = 2*k
    '''
    l = len(x)

    s = np.arange(-k, l - k)
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1

    f = np.zeros(k*2, dtype=np.float32)
    total = 0
    for i in range(-k, k):
        f[i+k] = np.exp(-(i/2)**2)
        total += f[i+k]
    f = f / total
    f = f.reshape(-1, 1)

    y = np.zeros(x.shape)
    for i in range(l):
        if e[i] - s[i] < 2*k:
            y[i] = np.mean(x[s[i]:e[i]], axis=0)
        else:
            y[i] = np.sum(x[s[i]:e[i]]*f, axis=0)
    return y

def compute_os_score(ground_truth, prediction):
    ground_truth_gbvn = ground_truth.groupby('video_id')
    label = ground_truth["label"].unique()
    scores = []
    for idx, this_pred in prediction.iterrows():
        video_id = this_pred['video_id']
        try:
            this_gt = ground_truth_gbvn.get_group(int(video_id))
            this_gt = this_gt.reset_index()
            tiou_arr = util_eval.segment_iou(this_pred[["start", "end"]].values, this_gt[["start", "end"]].values)
            scores += [item for item in tiou_arr if item > 0]
        except:
            print("Video {} gt has no {} action".format(video_id, label))
    return scores

class_names = ['Start', 'Stop', 'Slower', 'Faster', 'Done', 'FollowMe', 'Lift', 'Home', 'Interaction', \
              'Look', 'PickPart', "PositionPart", 'Report', "Ok", "Again", "Help", "Joystick", "Identification", 'Change']
class_map = dict([(str(i+1), class_names[i]) for i in range(len(class_names))])

def load_k_fold_probs(pickle_dir):
    probs = []
    with open(os.path.join(pickle_dir, "cobot_vmae_16x4.pkl"), "rb") as fp:
        vmae_16x4_probs = pickle.load(fp)
    probs.append(vmae_16x4_probs)
    return probs

def main():
    _FILENAME_TO_ID = {
    "s2_Alex_1":21,
    "s2_Alex_2":22,
    "s2_Alex_3":23,
    "s4_Gabin_1":41,
    "s4_Gabin_2":42,
    "s4_Gabin_3":43,
    "s6_LeVietDuc_1":61,
    "s6_LeVietDuc_2":62,
    "s6_LeVietDuc_3":63,
    "s8_PhamMinhHung_1":81,
    "s8_PhamMinhHung_2":82,
    "s8_PhamMinhHung_3":83,
    "s10_PhamQuangDai_1":101,
    "s10_PhamQuangDai_2":102,
    "s10_PhamQuangDai_3":103,
    "s12_NguyenXuanHieu_1":121,
    "s12_NguyenXuanHieu_2":122,
    "s12_NguyenXuanHieu_3":123,
    "s14_Alexandre_1":141,
    "s14_Alexandre_2":142,
    "s14_Alexandre_3":143,
    "s18_SungBin_1":181,
    "s18_SungBin_2":182,
    "s18_SungBin_3":183,
    "s20_TranHuyKhanh_1":201,
    "s20_TranHuyKhanh_2":202,
    "s20_TranHuyKhanh_3":203,
    "s22_MaiQuangTung_1":221,
    "s22_MaiQuangTung_2":222,
    "s22_MaiQuangTung_3":223,
    "s24_Nhung_1":241,
    "s24_Nhung_2":242,
    "s24_Nhung_3":243,
    "s26_DoThanhDat_1":261,
    "s26_DoThanhDat_2":262,
    "s26_DoThanhDat_3":263,
    "s28_CoBinh_1":281,
    "s28_CoBinh_2":282,
    "s28_CoBinh_3":283,
    "s30_ChiLich_1":301,
    "s30_ChiLich_2":302,
    "s30_ChiLich_3":303,
    "s32_DuyAnh_1":321,
    "s32_DuyAnh_2":322,
    "s32_DuyAnh_3":323,
    "s34_DoanNgocLinh_1":341,
    "s34_DoanNgocLinh_2":342,
    "s34_DoanNgocLinh_3":343,
    "s36_NguyenMaiChinh_1":361,
    "s36_NguyenMaiChinh_2":362,
    "s36_NguyenMaiChinh_3":363,
    "s38_PhamHuyHoang_1":381,
    "s38_PhamHuyHoang_2":382,
    "s38_PhamHuyHoang_3":383,
    "s40_LeHaHaiVan_1":401,
    "s40_LeHaHaiVan_2":402,
    "s40_LeHaHaiVan_3":403,
    "s42_TranPhuNghia_1":421,
    "s42_TranPhuNghia_2":422,
    "s42_TranPhuNghia_3":423,
    "s44_HaHuyenThu_1":441,
    "s44_HaHuyenThu_2":442,
    "s44_HaHuyenThu_3":443,
    "s46_TranTrungKien_1":461,
    "s46_TranTrungKien_2":462,
    "s46_TranTrungKien_3":463,
    "s48_VoHaiTrung_1":481,
    "s48_VoHaiTrung_2":482,
    "s48_VoHaiTrung_3":483,
    "s52_VuongVietHung_1":521,
    "s52_VuongVietHung_2":522,
    "s52_VuongVietHung_3":523,

    }
    classification = []
    localization = []
    pickle_dir = "pickles"
    k_flod_dash_probs = load_k_fold_probs(pickle_dir)
    for dash_vid in k_flod_dash_probs[0].keys():
        all_dash_probs = np.stack([np.array(list(map(np.array, dash_prob[dash_vid]))) for dash_prob in k_flod_dash_probs])
        avg_dash_seq = np.mean(all_dash_probs, axis=0)
        vid = _FILENAME_TO_ID[dash_vid]
        prob_seq = np.array(avg_dash_seq)
        prob_seq = np.squeeze(prob_seq)

        # activities_idx, startings, endings = activity_localization(prob_seq, action_threshold=0.1)
        # for label, s, e in zip(activities_idx, startings, endings):
        #     start = s * 30/30.
        #     end = e * 30/30
        #     classification.append([int(vid), label, start, end])

        prob_seq_smooth = smoothing(prob_seq, k=1) 

        activities_idx, startings, endings = activity_localization(prob_seq_smooth, action_threshold=0.1)
        for label, s, e in zip(activities_idx, startings, endings):
            start = s * 30/30.
            end = e * 30/30.
            localization.append([int(vid), label, start, end])

    # classification = pd.DataFrame(classification, columns =["video_id", "label", "start", "end"])
    rough_loc = pd.DataFrame(localization, columns =["video_id", "label", "start", "end"])
    prediction = general_submission(rough_loc)
    
    # classification.to_csv("cls.csv", columns =["video_id", "label", "start", "end"], index=False)
    # rough_loc.to_csv("rough_loc.csv", columns =["video_id", "label", "start", "end"], index=False)
    # prediction.to_csv("pred.csv", columns =["video_id", "label", "start", "end"], index=False)
    # load pred file
    CLASS_NUM = 19
    data_root = "./data"
    gt = pd.read_csv(os.path.join(data_root, "ground_truth.csv"))

    M, N = len(gt), len(prediction)
    gt_by_label = gt.groupby("label")
    pred_by_label = prediction.groupby("label")

    scores = []
    for label in range(1, CLASS_NUM+1):
        try:
            ground_truth_class = gt_by_label.get_group(label).reset_index(drop=True)
            prediction_class = pred_by_label.get_group(label).reset_index(drop=True)   
            scores += compute_os_score(ground_truth_class, prediction_class)
        except:
            continue
    print("Total Action:", M)
    print("True Positive:", len(scores))
    print("False Positive:", N-len(scores))
    print("False Negtive:", M-len(scores))
    print("score", sum(scores) / (M+N-len(scores)))
    print("PRECISON AVG", sum(scores)/len(scores))
    print("Recall AVG", sum(scores)/M)

main()