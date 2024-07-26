import csv
import os

A1_CLIP_PATH="./data/A1_clip"
folders = ['{}/{}'.format(A1_CLIP_PATH, i) for i in os.listdir(A1_CLIP_PATH)] # folder 0, 1, 2,...
file_name = []
train_files = []
test_files = []
for folder in folders:
    label = int(os.path.basename(folder))
    # file_name = ['{}/{}'.format(A1_CLIP_PATH, i) for i in os.listdir(folder)]
    for file in os.listdir(folder):
        idx = int(file.split("_")[0][1:]) #s0_, s1_... dùng chia train và test
        if idx%2 == 0:
            test_files.append(['{}/{}'.format(folder, file), label])
        else:
            train_files.append(['{}/{}'.format(folder, file), label])

with open("./train.csv", mode='w', newline="") as file:
    writer = csv.writer(file)
    writer.writerows(train_files)
    file.close()
print("done1")

with open("./test.csv", mode='w', newline="") as file1:
    writer = csv.writer(file1)
    writer.writerows(test_files)
    file1.close()
print("done2")