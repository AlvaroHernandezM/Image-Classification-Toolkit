from os import listdir, unlink
from os.path import join, dirname, abspath, exists
import os


def get_images(folder):
    return [('/' + join(folder, file), file.split('.')[0]) for file in listdir(folder)]


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
