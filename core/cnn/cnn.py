from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import model_from_json
from keras import backend as session
import numpy as np
from flask import jsonify


FOLDER_TRAINING_SET = 'core/cnn/dataset/training_set'
FOLDER_TEST_SET = 'core/cnn/dataset/test_set'
FILE_WEIGHTS_MODEL = 'core/cnn/models/model_weights.h5'
FILE_MODEL = 'core/cnn/models/model_cnn.json'
FILE_SINGLE_PREDICTION = 'core/cnn/dataset/single_prediction.jpg'
FILE_LABELS = 'core/cnn/models/labels.txt'
FILE_LOG = 'core/cnn/models/log.txt'


def train(steps_per_epoch=200, epochs=2, validation_steps=2000, positive_class='positive_class', negative_class='negative_class'):
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
    history_log = classifier.fit_generator(training_set,
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=epochs,
                                           validation_data=test_set,
                                           validation_steps=validation_steps)
    __write_file_log(history_log.history)
    __write_file_labels(positive_class, negative_class)
    return __save_model(classifier)


def classification():
    session.clear_session()
    with open(FILE_MODEL, 'r') as f:
        classifier = model_from_json(f.read())
    classifier.load_weights(FILE_WEIGHTS_MODEL)
    test_image = image.load_img(
        FILE_SINGLE_PREDICTION, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    file = open(FILE_LABELS, 'r')
    text_file = file.read()
    file.close()
    respond = {}
    if result[0][0] == 1:
        respond['cnn'] = str(text_file.split(';')[0])  # positive
    else:
        respond['cnn'] = str(text_file.split(';')[1])  # negative
    return jsonify(respond)


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


def __write_file_labels(positive_class, negative_class):
    file = open(FILE_LABELS, 'w')
    file.write(positive_class + ';' + negative_class)
    file.close()


def __write_file_log(history_log):
    file = open(FILE_LOG, 'w')
    file.write('val_loss:' + str(history_log['val_loss'][0]) + ';val_acc:' +
               str(history_log['val_acc'][0]) + ';loss:' + str(history_log['loss'][0]) + ';acc:' + str(history_log['acc'][0]))
    file.close()


if __name__ == '__main__':
    __write_file_log('uptc')
