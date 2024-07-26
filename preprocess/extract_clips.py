# Import everything needed to edit video clips
from moviepy.editor import VideoFileClip
import pandas as pd
import os
import PIL
import math
import argparse

parser = argparse.ArgumentParser("Get neccessay paths for running this file")
parser.add_argument('--a1_path', type=str, default='./data/A1')
parser.add_argument('--a1_clip_path', type=str, default="./data/A1_clip")
parser.add_argument('--ignore_file', type=str, default='./data/error_arm.csv')
parser.add_argument('--num_classes', type=int, default=20)

A1_PATH = parser.parse_args().a1_path
A1_CLIP_PATH = parser.parse_args().a1_clip_path
NUM_CLASSES = parser.parse_args().num_classes
ignore_file = parser.parse_args().ignore_file

def read_ignore_file(file):
    data = pd.read_csv(file, header=None)
    list_start = data.iloc[:, 2].values.tolist()
    list_subject = data.iloc[:, 0].values.tolist()
    list_subject = [str(i) for i in list_subject]
    return list_subject, list_start

def cut_video(clip1, time_start, time_end, path_video):
    # clip1 = VideoFileClip("test_phone.mp4").subclip(5, 18)
    clip1 = clip1.subclip(time_start, time_end)
    # getting width and height of clip 1
    w1 = clip1.w
    h1 = clip1.h
    
    print("Width x Height of clip 1 : ", end = " ")
    print(str(w1) + " x ", str(h1))
    
    print("---------------------------------------")
    
    # # resizing video downsize 50 %
    clip2 = clip1.resize((384, 384), PIL.Image.Resampling.LANCZOS)

    # getting width and height of clip 1
    w2 = clip2.w
    h2 = clip2.h
    
    print("Width x Height of clip 2 : ", end = " ")
    print(str(w2) + " x ", str(h2))
    
    print("---------------------------------------")
    clip2.write_videofile(path_video)

#create folder data
if not os.path.isdir(A1_CLIP_PATH):
    os.makedirs(A1_CLIP_PATH)
else: 
    print("folder already exists.")

for i in range(NUM_CLASSES):
    data_dir = '{}/{}'.format(A1_CLIP_PATH, str(i))
    CHECK_FOLDER = os.path.isdir(data_dir)
    if not CHECK_FOLDER:
        os.makedirs(data_dir)
    else:
        print(data_dir, "folder already exists.")
    print(i)

data_list = []
ignore_list_subject, ignore_list_start = read_ignore_file(ignore_file)
for folder_name in os.listdir(A1_PATH):
    path_folder = '{}/{}'.format(A1_PATH,folder_name)
    # path_csv = '{}/{}.csv'.format(path_folder, folder_name)
    list_video = ['{}/{}'.format(path_folder, i) for i in os.listdir(path_folder) if i.endswith(".mp4")]
    list_csv = [str(i).replace(".mp4", ".csv").replace("A1", "Annotation_v4") for i in list_video]
    for path_video, path_csv in zip(list_video, list_csv):
        print(path_folder, path_csv)
        df = pd.read_csv(path_csv)
        # filename_lst = list(df['Filename'].values) # có biến đổi tên cho cùng format
        filename_lst = str(os.path.basename(path_csv)).replace(".csv", "")
        label_lst = list(df['ID'].values) # dạng int
        start_time_lst = list(df['start'].values)
        end_time_lst = list(df['stop'].values)
        if filename_lst in ignore_list_subject:
            print('remove', filename_lst)
            index = ignore_list_subject.index(filename_lst)
            value = ignore_list_start[index]
            label_lst.pop(start_time_lst.index(value))
            end_time_lst.pop(start_time_lst.index(value))
            start_time_lst.remove(value)

        prev_file_name = ''
        file_name = ''
        count = 0
        count_nogesture = 0
        for i in range(len(label_lst)):
            file_name = filename_lst
            print('file_name', file_name)
            # print(count)
            count = 0

            if file_name != prev_file_name:
                # video_path = os.path.join(path_folder, file_name+".MP4")
                clip = VideoFileClip(path_video)
                clip_duration = int(clip.duration) # lấy độ dài clip theo giây
                clip_fps = int(clip.fps)
                prev_file_name = file_name
            # if label_lst[i].strip(" ").lstrip("Class") == "":
            #     continue
            clip_label = int(label_lst[i])+1 # để 0 cho no gesture
            time_start = round(start_time_lst[i]/clip_fps)
            time_end = round(end_time_lst[i]/clip_fps)
            clip_path = '{}/{}/{}_{}_{}.mp4'.format(A1_CLIP_PATH, clip_label, file_name, time_start, time_end)
            data_list.append([clip_path, clip_label])

            if not os.path.exists(clip_path):
                print("process {}".format(clip_path))
                cut_video(clip, time_start, time_end, clip_path) # cắt clip theo giây
            else:
                print("Already process {}".format(clip_path))
            if i == (len(df) - 1):
                print("Finished file {}".format(path_csv))
                break
        #Segment normal clip
            if count_nogesture <3:
                time_end = round(start_time_lst[i+1]/clip_fps)
                time_start = round(end_time_lst[i]/clip_fps)
                clip_path = '{}/{}/{}_{}_{}.MP4'.format(A1_CLIP_PATH, 0, file_name, time_start, time_end)
                data_list.append([clip_path, 0])

                if not os.path.exists(clip_path):
                    print("process {}".format(clip_path))
                    print("Clip duration", clip_duration, time_start, time_start)
                    time_end = min(time_end, clip_duration)
                    cut_video(clip, time_start, time_end, clip_path)
                    count_nogesture += 1
                else:
                    print("Already process {}".format(clip_path))
                count +=1
                    # break
                print(file_name)