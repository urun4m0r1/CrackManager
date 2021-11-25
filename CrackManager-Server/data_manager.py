import random
from os import path
from shutil import copy

import cv2
import keras
import numpy as np

from config_parser import Param, Path
from cv2_helper import get_all_images
from data_albumentator import get_preprocess, get_trn_augment, get_vld_augment
from file_manager import clean_tree, get_all_files


def split_train_valid(path_img, path_ann, path_data, retrain=False):
    ''' Make train valid dir from img, ann. '''

    files_list = list(zip(get_all_files(path_img), get_all_files(path_ann)))
    random.shuffle(files_list)

    paths_trn = [path_data + Path.TRN_IMG, path_data + Path.TRN_ANN]
    paths_vld = [path_data + Path.VLD_IMG, path_data + Path.VLD_ANN]
    clean_tree([*paths_trn, *paths_vld])

    ratio = Param.TRN_VLD_RATIO_RE_TRAIN if retrain else Param.TRN_VLD_RATIO
    
    num_trn = int(len(files_list) * ratio[0])
    files_trn = files_list[:num_trn]
    files_vld = files_list[num_trn:]

    for i in range(2):
        for file_trn in files_trn:
            copy(file_trn[i], paths_trn[i] + path.basename(file_trn[i]))
        for file_vld in files_vld:
            copy(file_vld[i], paths_vld[i] + path.basename(file_vld[i]))


class Dataset:
    ''' Read images, apply augmentation and preprocessing transformations. '''

    def __init__(self, imgs_dir, anns_dir, augmentation='none', preprocessing=False):
        self.imgs = get_all_images(imgs_dir)
        self.anns = get_all_images(anns_dir)
        assert len(self.imgs) == len(self.anns)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        img = cv2.cvtColor(self.imgs[i], cv2.COLOR_BGR2RGB)
        ann = cv2.cvtColor(self.anns[i], cv2.COLOR_BGR2RGB)

        h, w = ann.shape[:2]
        mask = np.zeros((h, w, 4))
        mask[:, :, 0] = np.all(ann == (255, 255, 255), axis=-1)  # TP, white
        mask[:, :, 1] = np.all(ann == (0, 255, 0), axis=-1)  # FN, green
        mask[:, :, 2] = np.all(ann == (255, 0, 0), axis=-1)  # FP, red
        mask[:, :, 3] = np.all(ann == (0, 0, 0), axis=-1)  # TN, black

        # apply augmentations
        if self.augmentation == 'train':
            augment = get_trn_augment()
            sample = augment(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
        elif self.augmentation == 'valid':
            augment = get_vld_augment()
            sample = augment(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
        elif self.augmentation == 'none':
            pass
        else:
            print("[EXCEPTION] Wrong augmentation name")

        # apply preprocessing
        if self.preprocessing:
            process = get_preprocess()
            sample = process(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        return img, mask

    def __len__(self):
        return len(self.imgs)


# Data generator for keras
class Dataloder(keras.utils.Sequence):
    ''' Load data from dataset and form batches. '''

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
