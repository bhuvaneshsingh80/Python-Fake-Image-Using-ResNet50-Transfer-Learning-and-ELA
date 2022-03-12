# Python-Fake-Image-Using-ResNet50-Transfer-Learning-and-ELA
How to detect Fake images using ResNet50 (Transfer Learning) and ELA.

MODEL:
RESNET50_WIDE

FEATURE ENGINEERING:
error-level analysis (ELA) 

HOW TO RUN:
INFERENCE FROM TRAINED WEIGHT:
python3 test.py -i <path to file>
EX: python3 test.py -i ./DATASET4/real/sandyB_real_64.jpg

TRAINING:
python3 train.py

PREPROCESSING:
python3 elm_prep_train_test_split.py

PLOTTING:
python3 plot.py

MediaEval Dataset - http://www.multimediaeval.org/mediaeval2016/
