from .cnn import cnn


def train(name, steps_per_epoch, epochs, validation_steps):
    if name == 'cnn':
        return cnn.train(steps_per_epoch, epochs, validation_steps)
