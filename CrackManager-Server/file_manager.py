from os import listdir, makedirs, path, unlink
from shutil import rmtree

from PyQt5.QtWidgets import QFileDialog


def get_all_files(tree, append_path=True):
    ''' Get all files in path. '''

    files = sorted([file for file in listdir(tree) if path.isfile(tree + file)])
    if files.__len__() == 0:
        raise FileNotFoundError
    else:
        return [path.join(tree, file) for file in files] if append_path else files


def get_newest_file(tree, append_path=True):
    ''' Get the newset file in path '''

    return max(get_all_files(tree, append_path), key=path.getctime)


def clean_tree(trees):
    ''' Safely remove dir and recreate '''

    for tree in trees:
        remove_tree(tree)
        if tree[-1] == '/':
            makedirs(tree, exist_ok=True)


def remove_tree(tree):
    ''' Safely remove tree '''

    if path.isdir(tree):
        rmtree(tree, True)
    elif path.isfile(tree) or path.islink(tree):
        unlink(tree)


def open_file_dialog(title, default_path, extensions):
    ''' Open file dialog and copy to result path '''

    filename = QFileDialog.getOpenFileName(None, title, default_path, extensions)
    if path.exists(filename[0]):
        return filename[0]
