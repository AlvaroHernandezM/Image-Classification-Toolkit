from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np

FOLDER_TRAINING_SET = 'core/cnn/dataset/training_set'
FOLDER_TEST_SET = 'core/cnn/dataset/test_set'
FILE_WEIGHTS_MODEL = 'core/cnn/models/model_weights.h5'
FILE_MODEL = 'core/cnn/models/model_cnn.json'
FILE_SINGLE_PREDICTION = 'core/cnn/dataset/single_prediction.jpg'


def train(steps_per_epoch, epochs=2, validation_steps=2000):
    classifier = Sequential()
    classifier = __build_architecture(classifier)
    classifier.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    training_set = train_datagen.flow_from_directory(FOLDER_TRAINING_SET,
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')
    test_set = test_datagen.flow_from_directory(FOLDER_TEST_SET,
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
    classifier.fit_generator(training_set,
                             steps_per_epoch=steps_per_epoch,
                             epochs=epochs,
                             validation_data=test_set,
                             validation_steps=validation_steps)
    return __save_model(classifier)


def classification(class_positive, class_negative):
    with open(FILE_MODEL, 'r') as f:
        classifier = model_from_json(f.read())
    classifier.load_weights(FILE_WEIGHTS_MODEL)
    test_image = image.load_img(
        FILE_SINGLE_PREDICTION, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    if result[0][0] == 1:
        return class_positive
    else:
        return class_negative


def __build_architecture(classifier):
    classifier.add(
        Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    return classifier


def __save_model(classifier):
    classifier.save_weights(FILE_WEIGHTS_MODEL)
    with open(FILE_MODEL, 'w') as f:
        f.write(classifier.to_json())
        return 'Modelo guardado!!!'
