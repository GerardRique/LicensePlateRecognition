import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]


def read_training_data(training_directory):
    image_data = []
    target_data = []
    for each_letter in letters:
        for each in range(10):
            image_path = os.path.join(training_directory, each_letter, each_letter + '_' + str(each) + '.jpg')
            # read each image of each character
            img_details = imread(image_path, as_gray=True)
            # converts each character image to binary image
            # binary_image = img_details < threshold_otsu(img_details)
            binary_image = img_details
            # the 2D array of each image is flattened because the machine learning
            # classifier requires that each sample is a 1D array
            # therefore the 20*20 image becomes 1*400
            # in machine learning terms that's 400 features with each pixel
            # representing a feature
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)

    return (np.array(image_data), np.array(target_data))




print('reading data')
training_dataset_dir = './train20X20'
image_data, target_data = read_training_data(training_dataset_dir)
print('reading data completed')


X = pd.DataFrame(image_data)
print(X.shape)
y = pd.DataFrame(target_data)
y = y[0]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

model = MLPClassifier(solver='lbfgs', activation='relu', learning_rate_init = 0.01, max_iter=400, alpha=1e-3, hidden_layer_sizes=(64,28), random_state=1)
model.fit(train_X, train_y)

# y.PattLen.value_counts().sort_index()


train_y_pred = model.predict(train_X)
test_y_pred = model.predict(test_X)

#Training Prediction Accuracy
print(accuracy_score(train_y.values, train_y_pred))
#Test Prediction Accuracy
print(accuracy_score(test_y.values, test_y_pred))

#Classification Report
print(classification_report(test_y.values, test_y_pred))

image_path = './train20X20/4/4_0.jpg'
img_details = imread(image_path, as_gray=True)

image_path_2 = './TrinidadLicensePlates/p/p_test3.jpg'
img_details_2 = imread(image_path_2, as_gray=True)

flat_bin_image = img_details.reshape(-1)
flat_bin_image_2 = img_details_2.reshape(-1)
my_input = []
my_input.append(flat_bin_image)
my_input.append(flat_bin_image_2)

result = model.predict(my_input)
print(result)

# import pickle
# print("model trained.saving model..")
# filename = './MLP_finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
# print("model saved")

