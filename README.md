# CS767-C1-Machine-Learning-Project : Automatic age and gender recognition system using real-world images -	By Priti Agrawal

As part of this project, I built an automated age group and gender classification system using real world images. The model will first detect and align faces in picture and then estimate age and gender using face features. 
I have used two machine learning technologies:
1. CNN using tensorflow
2. PCA with SVM

Dependencies
----------------------------------------------------------------
tensorflow ==1.14

scipy==1.1.0

numpy==1.15.4

opencv-python==3.4.4.19

tqdm==4.28.1

numpy == 1.13.3

pandas

dlib == 19.7.99

sklearn

Image Processing
----------------------------------------------------------------
For still images, I have used IMDB-WIKI dataset that consists of 500K+ face images with age and gender labels. For simplicity, you can download the pre-cropped imdb images (7GB) from below location:
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar

Run following command to perform image pre-processing:

python image_preprocess.py --db-path /Users/pritiagrawal/projectData/imdb_crop/imdb.mat \
--photo-dir /Users/pritiagrawal/projectData/imdb_crop \
--output-dir /Users/pritiagrawal/projectData \
--min-score 1.0 \
--img-size 224

Note: The image pre-processing stage will be common for both the following ML technologies

ML Technology-1 : CNN using tensorflow
-----------------------------------------------------------------
Train/Test/Evaluate model:

Run following command to train,test and evaluate the CNN model:

python model_training.py \
--img-dir /Users/pritiagrawal/projectData/crop \
--train-csv /Users/pritiagrawal/projectData/train.csv \
--val-csv /Users/pritiagrawal/projectData/val.csv \
--model-dir /Users/pritiagrawal/projectData \
--img-size 224 \
--num-steps 35000

Note: You can set the directory paths accordingly.

Predict age and gender of human face in new input images:

Run following command to make prediction:

python predict.py \
--mat-path /Users/pritiagrawal/projectData/imdb_crop/imdb.mat  \
--model-dir /Users/pritiagrawal/projectData/serving/1555814019 \
--project-path /Users/pritiagrawal/projectData 


ML Technology-2 : PCA with SVM
-----------------------------------------------------------------
Train/Test model:

Run following command to train and test SVM model(using PCA for feature extraction):

python pca_svm.py \
--train-csv /Users/pritiagrawal/projectData/train.csv \
--img-dir /Users/pritiagrawal/projectData/crop





















