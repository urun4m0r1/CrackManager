import csv

import cv2
import numpy as np

from config_parser import Param, Path
from cv2_helper import dliatie_crack, get_all_images, make_overlay
from feature_matcher import apply_homography, make_homography
from file_manager import clean_tree
from gui_helper import TaskTime


def __test_feature_matcher():
    ''' Feature matcher test code '''

    ALGORITHMS = ['akaze', 'brisk_fast', 'brisk_slow', 'orb_fast', 'orb_slow']
    NUM_DIST = 2
    NUM_WALL = 9
    NUM_ITER = 3

    print(">>> Setting up images ...")
    imgs_query = get_all_images(Path.SAMPLES + 'wall/')
    imgs_crack = get_all_images(Path.SAMPLES + 'crack/')
    imgs_trains = [[cv2.imread(Path.SAMPLES + f'holo/holo_{i+1}_{j+1}.jpg')
                    for j in range(NUM_WALL)] for i in range(NUM_DIST)]
    imgs_maskgts = [[cv2.imread(Path.SAMPLES + f'maskgt/maskgt_{i+1}_{j+1}.png')
                     for j in range(NUM_WALL)] for i in range(NUM_DIST)]

    clean_tree([Path.SAMPLES + 'match/', Path.SAMPLES + 'overlay/', Path.SAMPLES + 'mask/'])

    logs_path = Path.SAMPLES + 'feature_match_logs.csv'
    with open(logs_path, 'w', newline='') as logs:
        writer = csv.writer(logs)
        writer.writerow(['alogrithm', 'distance', 'wall', 'features', 'mean_times', 'stdev_times', 'errors'])
    for alogrithm in ALGORITHMS:
        for i, (imgs_train, imgs_maskgt) in enumerate(zip(imgs_trains, imgs_maskgts)):
            for j, (img_query, img_crack, img_train, img_maskgt) in enumerate(zip(imgs_query, imgs_crack, imgs_train, imgs_maskgt)):
                print("------------------------------------")
                features = 0
                times = [0.0] * NUM_ITER
                errors = 0
                for k in range(NUM_ITER):
                    task = TaskTime(f"featureMatch_[{alogrithm}]_[dist={i+1}]_[wall={j+1}]_[iter={k+1}]")
                    features, img_match, affine_info = make_homography(
                        img_query, img_train, alogrithm, Param.EQUALIZER)
                    if features:
                        img_crack = dliatie_crack(img_crack, Param.DILATE)
                        img_overlay = make_overlay(img_train, apply_homography(img_crack, affine_info))
                        img_mask = apply_homography(img_crack, affine_info, make_mask=True)
                        px_intersect = np.sum(cv2.bitwise_and(img_maskgt[:, :, :3], img_mask[:, :, :3]))
                        px_union = np.sum(cv2.bitwise_or(img_maskgt[:, :, :3], img_mask[:, :, :3]))
                        errors = abs(1 - (px_union / px_intersect)) * 100
                        times[k] = task.display_time()
                    else:
                        times[k] = task.display_time()
                        break

                with open(logs_path, 'a', newline='') as logs:
                    writer = csv.writer(logs)
                    writer.writerow([alogrithm, i+1, j+1, features, np.mean(times), np.std(times), errors])

                if features > 0:
                    cv2.imwrite(Path.SAMPLES + f'match/match_{alogrithm}_{i+1}_{j+1}.png', img_match)
                    cv2.imwrite(Path.SAMPLES + f'overlay/overlay_{alogrithm}_{i+1}_{j+1}.png', img_overlay)
                    cv2.imwrite(Path.SAMPLES + f'mask/mask_{alogrithm}_{i+1}_{j+1}.png', img_mask)


if __name__ == '__main__':
    __test_feature_matcher()
