from .cnn import cnn
from .image_retraining import image_retraining
from .svm_knn_bpnn import svm_knn_bpnn


def train_cnn(steps_per_epoch, epochs, validation_steps, positive_class, negative_class):
    return cnn.train(steps_per_epoch, epochs, validation_steps, positive_class, negative_class)


def classification_cnn():
    return cnn.classification()


def train_image_retraining(training_steps):
    return image_retraining.train(training_steps)


def classification_image_retraining():
    return image_retraining.classification()


def train_svm_knn_bpnn(number_neighbors, focus, hidden_layer_sizes, max_iter_bpnn, max_iter_svm):
    return svm_knn_bpnn.train(number_neighbors, focus, hidden_layer_sizes, max_iter_bpnn, max_iter_svm)


def classification_svm_knn_bpnn():
    return svm_knn_bpnn.classification()
