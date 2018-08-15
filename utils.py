from os import listdir, unlink
from os.path import join, dirname, abspath, exists
import os


def get_images(folder):
    return [('/' + join(folder, file), file.split('.')[0]) for file in listdir(folder)]


def delete_images(folder):
    [unlink(dirname(abspath(file)) + '/' + join(folder, file)) for file in listdir(folder)]


def count_folders(folder):
    if os.path.exists(folder):
        return len(listdir(folder))
    else:
        return 0


def create_folder(folder):
    if not os.path.exists(folder):
        os.system('mkdir ' + folder)
    else:
        return 'folder-exists'


def move_images(origen, destination):
    if not os.path.exists(destination):
        create_folder(destination)
    os.system('cp ' + origen + '/* ' + destination)
