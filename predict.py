import logging
from argparse import ArgumentParser
from pathlib import Path
import tensorflow as tf
import os
from tensorflow.contrib import predictor
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from datetime import datetime
import dlib
from tqdm import tqdm
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tf.logging.set_verbosity(tf.logging.INFO)


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1

# Define predicted age group
def predict_age_range(output_age):
    predicted_age = ''
    if output_age == 0:
        predicted_age = '0-3'
    elif output_age == 1:
        predicted_age = '4-7'
    elif output_age == 2:
        predicted_age = '8-14'
    elif output_age == 3:
        predicted_age = '15-24'
    elif output_age == 4:
        predicted_age = '25-32'
    elif output_age == 5:
        predicted_age = '33-43'
    elif output_age == 6:
        predicted_age = '44-59'
    else:
        predicted_age = '60+'
    return predicted_age


if __name__ == '__main__':
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--mat-path', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--project-path', required=True)

    args = parser.parse_args()

    db = loadmat(args.mat_path)['imdb'][0, 0]

    # model_dir=r'/Users/pritiagrawal/projectData/serving/1555699129'
    model_dir= args.model_dir
    prediction_fn = predictor.from_saved_model(export_dir=model_dir, signature_def_key='serving_default')

    # dataset_root= r'/Users/pritiagrawal/projectData/'
    dataset_root =args.project_path
    validation_image_dir = Path(os.path.join(dataset_root,"predict_image"))
    image_paths = list(validation_image_dir.glob("*.jpg"))

    n_row,n_col=3,4
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

    for i, image_path in tqdm(enumerate(image_paths)):

        for j in range(len(db["face_score"][0])):
            img_name =str(db["full_path"][0][j][0]).rsplit('/',1)[1]
            actual_gender =2
            actual_age = ''
            if img_name==str(image_path).rsplit('/',1)[1]:
                dob = db["dob"][0][j]  # Matlab serial date number
                gender = db["gender"][0][j]
                age = calc_age(db["photo_taken"][0][j], dob)
                location = db["face_location"]
                actual_gender = int(gender)
                actual_age = age
                break


        image = cv2.imread(str(image_path))

        image = cv2.resize(image,(224,224))
        output = prediction_fn({
            'image': [image]
        })


        predicted_age = predict_age_range(output['age_class'][0])

        detector = dlib.get_frontal_face_detector()

        detected = detector(image, 1)
        faces = np.empty((len(detected), 224, 224, 3))
        img_h, img_w, _ = np.shape(image)

        if actual_gender == 1:
            actual_gender = 'M'
        elif actual_gender==0:
            actual_gender='F'
        else:
            actual_gender=''

        if output['classes'][0] == 1:
            predicted_gender = 'M'
        elif output['classes'][0]==0:
            predicted_gender='F'


        for k, d in tqdm(enumerate(detected)):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("A:{0},{1} P:{2},{3}".format(actual_age,actual_gender,predicted_age,predicted_gender))
        plt.xticks(())
        plt.yticks(())
    plt.show()