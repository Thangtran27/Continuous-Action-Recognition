## Installing the environments
Please follow the installation instructions in [VideoMAE](https://github.com/MCG-NJU/VideoMAE). Also, you can simply run the following command:
```
conda env create -f requirements.yaml
conda activate env_test39
```
## Data Preprocessing
**1. Data download**
Download the SigMGesture dataset including "video" and "Annotation_v4" and then put it into "./data/A1" and "./data" respectively.

**2. Action segments extraction**
To slit data into clip for each gesture, run the following command and change paths if need be:

```
python preprocess/extract_clips.py
```
and the video segments will be saved in folder "data/A1_clip".

## Training model
Model is initialized with pretrained VideoMAE with  Kinetic-710, you can get the pretrained weights from [VideoMAE-Kinetic-710/ViT-L](https://drive.google.com/file/d/1jX1CiqxSkCfc94y8FRW1YGHy-GNvHCuD/view?usp=sharing). Next, put it into "./data" folder.
Change paths in the "scripts/cls/train_cls.sh" file if need be. Then run this command:
```
bash scripts/cls/train_cls.sh
```
## Inference model
Run the following command:
```inference 
bash scripts/cls/inference_cls.sh 
```
After completing the process, Top1 Accuracy is printed into the screen

## Postprocessing
To get the final TAL results, you need to perform postprocessing on  
```
python postprocessing.py
```
To get Frame-wise Accuracy result, run the following command and check paths if need be:
```
python frame_wise_acc.py
```
