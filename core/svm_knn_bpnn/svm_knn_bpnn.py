
# python trainModels.py --dataset "dataset" --neighbors "10"
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import imutils
import cv2
import os
import time

FOLDER_DATASET = 'dataset/'
FOLDER_MODELS = 'models/'
FOLDER_MODELS_HISTOGRAM = FOLDER_MODELS + 'histogram/'
FILE_MODELS_HISTOGRAM_KNN = FOLDER_MODELS_HISTOGRAM + 'knn.pkl'
FILE_MODELS_HISTOGRAM_SVM = FOLDER_MODELS_HISTOGRAM + 'svm.pkl'
FILE_MODELS_HISTOGRAM_BPNN = FOLDER_MODELS_HISTOGRAM + 'bpnn.pkl'
FOLDER_MODELS_PIXEL = FOLDER_MODELS + 'pixel/'
FILE_MODELS_PIXEL_KNN = FOLDER_MODELS_PIXEL + 'knn.pkl'
FILE_MODELS_PIXEL_SVM = FOLDER_MODELS_PIXEL + 'svm.pkl'
FILE_MODELS_PIXEL_BPNN = FOLDER_MODELS_PIXEL + 'bpnn.pkl'
FILE_LOG = 'models/log.txt'


def train(number_neighbors, focus, hidden_layer_sizes, max_iter_bpnn, max_iter_svm):
    __init_log()
    imagePaths = list(paths.list_images(FOLDER_DATASET))

    rawImages = []
    features = []
    labels = []

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        pixels = __image_to_feature_vector(image)
        hist = __extract_color_histogram(image)

        rawImages.append(pixels)
        features.append(hist)
        labels.append(label)

        if i > 0 and ((i + 1) % 200 == 0 or i == len(imagePaths) - 1):
            line = "[INFO] processed {}/{}".format(i + 1, len(imagePaths))
            __write_log(line)

    labels = np.array(labels)
    if focus == 'histogram':
        features = np.array(features)
        line = "[INFO] features matrix: {:.2f}MB".format(
            features.nbytes / (1024 * 1000.0))
        __write_log(line)
        (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
            features, labels, test_size=0.15, random_state=42)

        # k-NN
        line = "[INFO] evaluating histogram accuracy..."
        __write_log(line)
        model = KNeighborsClassifier(n_neighbors=number_neighbors)
        model.fit(trainFeat, trainLabels)
        acc = model.score(testFeat, testLabels)
        line = "[INFO] k-NN classifier: k=%d" % number_neighbors
        __write_log(line)
        line = "[INFO] histogram accuracy: {:.2f}%".format(acc * 100)
        __write_log(line)
        joblib.dump(model, FILE_MODELS_HISTOGRAM_KNN)

        # neural network
        line = "[INFO] evaluating histogram accuracy..."
        __write_log(line)
        model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
                              solver='sgd', tol=1e-4, random_state=1,
                              learning_rate_init=.1)
        model.fit(trainFeat, trainLabels)
        acc = model.score(testFeat, testLabels)
        line = "[INFO] neural network histogram accuracy: {:.2f}%".format(acc * 100)
        __write_log(line)
        joblib.dump(model, FILE_MODELS_HISTOGRAM_BPNN)

        # SVC
        line = "[INFO] evaluating histogram accuracy..."
        __write_log(line)
        model = SVC(max_iter=1000, class_weight='balanced')
        model.fit(trainFeat, trainLabels)
        acc = model.score(testFeat, testLabels)
        line = "[INFO] SVM-SVC histogram accuracy: {:.2f}%".format(acc * 100)
        __write_log(line)
        joblib.dump(model, FILE_MODELS_HISTOGRAM_SVM)

        return 'Modelos guardados!!!'
    elif focus == 'pixel':
        rawImages = np.array(rawImages)
        line = "[INFO] pixels matrix: {:.2f}MB".format(
            rawImages.nbytes / (1024 * 1000.0))
        __write_log(line)
        (trainRI, testRI, trainRL, testRL) = train_test_split(
            rawImages, labels, test_size=0.15, random_state=42)

        # k-NN
        line = "[INFO] evaluating raw pixel accuracy..."
        __write_log(line)
        model = KNeighborsClassifier(n_neighbors=number_neighbors)
        model.fit(trainRI, trainRL)
        acc = model.score(testRI, testRL)
        line = "[INFO] k-NN classifier: k=%d" % number_neighbors
        __write_log(line)
        line = "[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100)
        __write_log(line)
        joblib.dump(model, FILE_MODELS_PIXEL_KNN)

        # neural network
        line = "[INFO] evaluating raw pixel accuracy..."
        __write_log(line)
        model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), max_iter=max_iter_bpnn, alpha=1e-4,
                              solver='sgd', tol=1e-4, random_state=1,
                              learning_rate_init=.1)
        model.fit(trainRI, trainRL)
        acc = model.score(testRI, testRL)
        line = "[INFO] neural network raw pixel accuracy: {:.2f}%".format(
            acc * 100)
        __write_log(line)
        joblib.dump(model, FILE_MODELS_PIXEL_BPNN)

        # SVC
        line = "[INFO] evaluating raw pixel accuracy..."
        __write_log(line)
        model = SVC(max_iter=max_iter_svm, class_weight='balanced')
        model.fit(trainRI, trainRL)
        acc = model.score(testRI, testRL)
        line = "[INFO] SVM-SVC raw pixel accuracy: {:.2f}%".format(acc * 100)
        __write_log(line)
        joblib.dump(model, FILE_MODELS_PIXEL_SVM)
        return 'Modelos guardados!!!'
    else:
        return 'error: focus no esperado'


def __write_log(line):
    file = open(FILE_LOG, 'a+')
    file.write(str(line) + '\n')
    file.close()


def __init_log():
    file = open(FILE_LOG, 'w')
    file.write(time.strftime("%c") + '\n')
    file.close()


def __image_to_feature_vector(image, size=(128, 128)):
    return cv2.resize(image, size).flatten()


def __extract_color_histogram(image, bins=(32, 32, 32)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return hist.flatten()


if __name__ == '__main__':
    print (train(2, 'histogram', 50, 200, 200))
