from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import cv2
import numpy as np
import re
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_learning_curves
import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser(add_help=True)
parser.add_argument('--train-csv', required=True)
parser.add_argument('--img-dir', required=True)

args = parser.parse_args()

def preprocess(im):
    im = cv2.resize(im, (100, 100))
    return cv2.equalizeHist(im)

f = open(args.train_csv,"r")
c=0
x=[]
y=[]
first_line = f.readline()
first_line = f.readline()
for i in f:
    if not(re.match(r'^\s*$', i)):
        filename = i.split(',')[0]
        trainfile=os.path.abspath(os.path.join(args.img_dir, filename))
        img = cv2.imread(trainfile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x.append(img)
        y.append(i[-2])
        if c ==2000:
            break
        c=c+1
images = (preprocess(im) for im in x)
X = np.vstack([im.flatten() for im in images])
Y = np.hstack(y)
print(X.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.1,random_state=100)
pca = decomposition.PCA(n_components=25, svd_solver='randomized',
          whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((25,100,100))
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
clf = clf.fit(X_train_pca, Y_train)

y_pred = clf.predict(X_test_pca)


print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred, labels=['0','1']))

# Plot the learning curves
plt.figure(figsize=(20,10))
plot_learning_curves(X_train_pca, Y_train, X_test_pca, Y_test, clf,scoring="accuracy")
plt.title("Learning Curves")

plt.show()

# plot the result of the prediction on a portion of the test set
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[int(y_pred[i])]
    true_name = target_names[int(y_test[i])]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
print(Y_test)
prediction_titles = [title(y_pred, Y_test, [0,1], i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, 100, 100)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, 100, 100)

plt.show()

