import os
from pathlib import Path
def key_sort(file):
    basename = os.path.basename(file)
    first = str(basename).split("_")[0][1:]
    last = str(basename).split(".")[0][-1]
    number = int(first + last)
    return number

def test_filter(file_path):
    basename = os.path.basename(file_path)
    person_idx = str(basename).split("_")[0][1:]
    if int(person_idx) %2 == 0:
        return True
    else:
        return False

def get_path(label_path, video_cut):
    label_list = []
    for file in Path(label_path).iterdir(): # D:\MATERIAL\COBOT\New_project\Sperical_coor\pose_legacy\s10_PhamQuangDai_1.npy
        folder_list = os.listdir(file.as_posix())
        folder_list = ["{}/{}".format(file.as_posix(), i) for i in folder_list if i.endswith(".mp4")]
        label_list += folder_list #D:/MATERIAL/COBOT/New_project/Sperical_coor/pose_legacy/s10_PhamQuangDai_1.npy
        label_list.sort(key=key_sort)
    label_list = [i for i in label_list if os.path.basename(i) not in video_cut]
    return label_list

def get_video_cut(path):
    video_path = ["{}/{}".format(path, i) for i in os.listdir(path) if i.endswith(".mp4")]
    basename = [i for i in os.listdir(path) if i.endswith(".mp4")]
    return video_path, basename

video_cut, basename = get_video_cut("/home/vnpt/THANGTV/models/checkpoints/aicity_release/data/clip_error_data")
path = get_path("/home/vnpt/THANGTV/models/checkpoints/aicity_release/data/A1", basename)
video_path = list(filter(test_filter, path))
video_path += video_cut
with open("data/video_ids_cut.txt", "w") as f:
    for i in video_path:
        f.write(i+"\n")