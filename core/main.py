from .cnn import cnn


def train_cnn(steps_per_epoch, epochs, validation_steps, positive_class, negative_class):
    return cnn.train(steps_per_epoch, epochs, validation_steps, positive_class, negative_class)


def clasification_cnn():
    return cnn.classification()
