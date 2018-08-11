from .cnn import cnn
from .image_retraining import image_retraining


def train_cnn(steps_per_epoch, epochs, validation_steps, positive_class, negative_class):
    return cnn.train(steps_per_epoch, epochs, validation_steps, positive_class, negative_class)


def classification_cnn():
    return cnn.classification()


def train_image_retraining():
    return image_retraining.train()


def classification_image_retraining():
    return image_retraining.classification()
