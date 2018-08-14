
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
from flask import jsonify


FOLDER_DATASET = 'core/svm_knn_bpnn/dataset/'
FOLDER_MODELS = 'core/svm_knn_bpnn/models/'
FOLDER_MODELS_HISTOGRAM = FOLDER_MODELS + 'histogram/'
FILE_MODELS_HISTOGRAM_KNN = FOLDER_MODELS_HISTOGRAM + 'knn.pkl'
FILE_MODELS_HISTOGRAM_SVM = FOLDER_MODELS_HISTOGRAM + 'svm.pkl'
FILE_MODELS_HISTOGRAM_BPNN = FOLDER_MODELS_HISTOGRAM + 'bpnn.pkl'
FOLDER_MODELS_PIXEL = FOLDER_MODELS + 'pixel/'
FILE_MODELS_PIXEL_KNN = FOLDER_MODELS_PIXEL + 'knn.pkl'
FILE_MODELS_PIXEL_SVM = FOLDER_MODELS_PIXEL + 'svm.pkl'
FILE_MODELS_PIXEL_BPNN = FOLDER_MODELS_PIXEL + 'bpnn.pkl'
FILE_LOG = 'core/svm_knn_bpnn/models/log.txt'
FILE_SINGLE_PREDICTION = 'core/svm_knn_bpnn/single_prediction/single_prediction.jpg'


def classification():
    imagePost = cv2.imread(FILE_SINGLE_PREDICTION)
    respond = {}
    focus = __read_focus()
    respond['focus'] = focus
    if focus == 'histogram':
        histimagePost = __extract_color_histogram(imagePost)
        features = []
        features.append(histimagePost)
        # k-NN
        model = joblib.load(FILE_MODELS_HISTOGRAM_KNN)
        respond['knn'] = model.predict(features)[0]
        # neural network
        model = joblib.load(FILE_MODELS_HISTOGRAM_BPNN)
        respond['bpnn'] = model.predict(features)[0]
        # SVC
        model = joblib.load(FILE_MODELS_HISTOGRAM_SVM)
        respond['svm'] = model.predict(features)[0]

    elif focus == 'pixel':
        pixelsimagePost = __image_to_feature_vector(imagePost)
        rawImages = []
        rawImages.append(pixelsimagePost)
        # k-NN
        model = joblib.load(FILE_MODELS_PIXEL_KNN)
        respond['knn'] = model.predict(rawImages)[0]
        # neural network
        model = joblib.load(FILE_MODELS_PIXEL_BPNN)
        respond['bpnn'] = model.predict(rawImages)[0]
        # SVC
        model = joblib.load(FILE_MODELS_PIXEL_SVM)
        respond['svm'] = model.predict(rawImages)[0]
    else:
        respond['status'] = 'error'
    return jsonify(respond)


def train(number_neighbors, focus, hidden_layer_sizes, max_iter_bpnn, max_iter_svm):
    respond = {}
    __init_log(focus)
    respond['focus'] = focus
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
            size_images = len(imagePaths)
            line = "[INFO] processed {}/{}".format(i + 1, size_images)
            respond['length_images'] = size_images
            __write_log(line)

    labels = np.array(labels)
    if focus == 'histogram':
        features = np.array(features)
        size_package = '{:.2f}MB'.format(
            features.nbytes / (1024 * 1000.0))
        line = '[INFO] features matrix: ' + size_package
        respond['size_package'] = size_package
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
        accuracy_knn = '{:.2f}%'.format(acc * 100)
        line = "[INFO] histogram accuracy: " + accuracy_knn
        respond['accuracy_knn'] = accuracy_knn
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
        accuracy_bpnn = "{:.2f}%".format(acc * 100)
        line = "[INFO] neural network histogram accuracy: " + accuracy_bpnn
        respond['accuracy_bpnn'] = accuracy_bpnn
        __write_log(line)
        joblib.dump(model, FILE_MODELS_HISTOGRAM_BPNN)

        # SVC
        line = "[INFO] evaluating histogram accuracy..."
        __write_log(line)
        model = SVC(max_iter=1000, class_weight='balanced')
        model.fit(trainFeat, trainLabels)
        acc = model.score(testFeat, testLabels)
        accuracy_svm = '{:.2f}%'.format(acc * 100)
        line = "[INFO] SVM-SVC histogram accuracy: " + accuracy_svm
        respond['accuracy_svm'] = accuracy_svm
        __write_log(line)
        joblib.dump(model, FILE_MODELS_HISTOGRAM_SVM)
        respond['success'] = 'true'
        return jsonify(respond)
    elif focus == 'pixel':
        rawImages = np.array(rawImages)
        size_package = "{:.2f}MB".format(
            rawImages.nbytes / (1024 * 1000.0))
        line = "[INFO] pixels matrix: " + size_package
        respond['size_package'] = size_package
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
        accuracy_knn = "{:.2f}%".format(acc * 100)
        line = "[INFO] raw pixel accuracy: " + accuracy_knn
        respond['accuracy_knn'] = accuracy_knn
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
        accuracy_bpnn = "{:.2f}%".format(acc * 100)
        line = "[INFO] neural network raw pixel accuracy: " + accuracy_bpnn
        respond['accuracy_bpnn'] = accuracy_bpnn
        __write_log(line)
        joblib.dump(model, FILE_MODELS_PIXEL_BPNN)

        # SVC
        line = "[INFO] evaluating raw pixel accuracy..."
        __write_log(line)
        model = SVC(max_iter=max_iter_svm, class_weight='balanced')
        model.fit(trainRI, trainRL)
        acc = model.score(testRI, testRL)
        accuracy_svm = "{:.2f}%".format(acc * 100)
        line = "[INFO] SVM-SVC raw pixel accuracy: " + accuracy_svm
        respond['accuracy_svm'] = accuracy_svm
        __write_log(line)
        joblib.dump(model, FILE_MODELS_PIXEL_SVM)
        respond['success'] = 'true'
        return jsonify(respond)
    else:
        respond['success'] = 'false'
        respond['error'] = 'focus no esperado'
        return jsonify(respond)


def __read_focus():
    file = open(FILE_LOG, 'r')
    text_log = file.read()
    file.close()
    return text_log.split(';')[0]


def __write_log(line):
    file = open(FILE_LOG, 'a+')
    file.write(str(line) + '\n')
    file.close()


def __init_log(focus):
    file = open(FILE_LOG, 'w')
    file.write(focus + ';' + time.strftime("%c") + '\n')
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
    #print (train(2, 'histogram', 50, 200, 200))
    print (classification())
