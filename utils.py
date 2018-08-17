from os import listdir, unlink
from os.path import join, dirname, abspath, exists
import os


def get_images(folder):
    return [join(folder, file) for file in listdir(folder)]


def delete_images(folder):
    [unlink(dirname(abspath(file)) + '/' + join(folder, file)) for file in listdir(folder)]
    return 'delete-images-success'


def count_folders(folder):
    if os.path.exists(folder):
        return len(listdir(folder))
    else:
        return 0


def create_folder(folder):
    if not os.path.exists(folder):
        os.system('mkdir ' + folder)
        return 'folder-create-success'
    else:
        return 'folder-exists'


def move_images(origen, destination):
    if not os.path.exists(destination):
        create_folder(destination)
    os.system('cp ' + origen + '* ' + destination)
    return 'images-move-success'


def move_images_image_retraining(folder_image_retraining, folder_positive, folder_negative, class_negative, class_positive):
    if create_folder(folder_image_retraining + class_positive) == 'folder-create-success' or count_folders(folder_image_retraining + class_positive) == 0:
        move_images(folder_positive, folder_image_retraining + class_positive)
    if create_folder(folder_image_retraining + class_negative) == 'folder-create-success' or count_folders(folder_image_retraining + class_negative) == 0:
        move_images(folder_negative, folder_image_retraining + class_negative)


def move_images_cnn(folder_positive, folder_negative, folder_negative_training_dataset_cnn, folder_positive_training_dataset_cnn, folder_negative_test_dataset_cnn, folder_positive_test_dataset_cnn):
    images_positive = get_images(folder_positive)
    images_negative = get_images(folder_negative)
    number_percentaje_test_positive = round(len(images_positive) * 0.15)
    number_percentaje_test_negative = round(len(images_negative) * 0.15)
    i = 1
    for image_positive in images_positive:
        if i <= number_percentaje_test_positive:
            move_image(image_positive, folder_positive_test_dataset_cnn)
        else:
            move_image(image_positive, folder_positive_training_dataset_cnn)
        i = i + 1
    i = 1
    for image_negative in images_negative:
        if i <= number_percentaje_test_negative:
            move_image(image_negative, folder_negative_test_dataset_cnn)
        else:
            move_image(image_negative, folder_negative_training_dataset_cnn)
        i = i + 1
    return 'images-move-success'


def move_images_svm_knn_bpnn(folder_positive, folder_negative, folder_svm_knn_bpnn, class_positive, class_negative):
    images_positive = get_images(folder_positive)
    images_negative = get_images(folder_negative)
    for image_positive in images_positive:
        parts = image_positive.split('/')
        filename = folder_svm_knn_bpnn + class_positive + '.' + parts[len(parts) - 1]
        os.system('cp ' + image_positive + ' ' + filename)
    for image_negative in images_negative:
        parts = image_negative.split('/')
        filename = folder_svm_knn_bpnn + class_negative + '.' + parts[len(parts) - 1]
        os.system('cp ' + image_negative + ' ' + filename)
    return 'images-move-success'


def move_image(url_image, folder_destination):
    if not os.path.exists(folder_destination):
        create_folder(folder_destination)
    os.system('cp ' + url_image + ' ' + folder_destination)
    return 'image-move-success'


if __name__ == "__main__":
    print('main')
