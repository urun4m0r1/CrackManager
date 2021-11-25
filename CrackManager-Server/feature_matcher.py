import math

import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

from config_parser import Param
from cv2_helper import (change_color, fill_edge, imresize, match_histogram,
                        match_resize, dliatie_crack)
from gui_helper import TaskTime


def __getAngle(a, b, c):
    ''' Get angle between therr points. '''

    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def __validate_edges(edges):
    ''' Check points are align in clockwise direction and has rectangular shape. '''

    a, b, c, d = edges[:, 0]
    ang_a = __getAngle(d, a, b)
    ang_b = __getAngle(a, b, c)
    ang_c = __getAngle(b, c, d)
    ang_d = __getAngle(c, d, a)
    angs = [ang_a, ang_b, ang_c, ang_d]

    return (a[1] < b[1] and b[0] < c[0] and c[1] > d[1] and d[0] > a[0]) and not (True in [ang > 180 for ang in angs])


def __format_cluster_info(index, match):
    ''' Get cluster info. '''

    return f"{index}: {match}/{Param.MIN_MATCH}"


def apply_homography(img, affine_info, make_mask=False):
    ''' Apply homography to crack image. '''

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = dliatie_crack(img, Param.DILATE)
    img = change_color(img, (0, 0, 255), (0, 0, 0))
    img = change_color(img, (0, 255, 0), (255, 255, 255))

    h, w = affine_info[1]
    #height, width = img.shape[:2]
    #height = int(width * (h / w))
    img = imresize(img, w, h)

    # Apply homograpy with affine info
    img = fill_edge(img, w if make_mask else Param.EDGE, 0)
    img = cv2.warpPerspective(img, affine_info[0], (w, h))
    img = fill_edge(img, 0 if make_mask else Param.EDGE_CUT, 0)

    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, tmp = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    for i in range(3):
        _, img[..., i] = cv2.threshold(img[..., i], 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, tmp]
    img = cv2.merge(rgba, 4)
    img = change_color(img, (255, 255, 255), (255, 255, 255) if make_mask else Param.COLOR_CRACK)
    img[:, :, 3] -= 0 if make_mask else Param.TRANSPARENCY
    
    return img


def make_homography(img_query, img_train, detector_name, equalizer):
    ''' Feature matcher for img_query and img_train image. '''

    # Setting up images and feature matcher
    img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
    img_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2GRAY)
    img_query, img_train = match_resize(img_query, img_train)

    if equalizer == 'equalize':
        img_query = cv2.equalizeHist(img_query)
        img_train = cv2.equalizeHist(img_train)
    elif equalizer == 'match':
        img_query, img_train = match_histogram(img_query, img_train)
    elif equalizer == 'none':
        pass
    else:
        print("[EXCEPTION] Wrong equalizer name")

    if detector_name == 'akaze':
        detector = cv2.AKAZE_create()
    elif detector_name == 'brisk_slow':
        detector = cv2.BRISK_create(Param.BRISK_THRESH_SLOW)
    elif detector_name == 'brisk_fast':
        detector = cv2.BRISK_create(Param.BRISK_THRESH_FAST)
    elif detector_name == 'orb_fast':
        detector = cv2.ORB_create(Param.ORB_FEATURES_FAST)
    elif detector_name == 'orb_slow':
        detector = cv2.ORB_create(Param.ORB_FEATURES_SLOW)
    else:
        print("[EXCEPTION] Wrong feature detector name")

    # Detecting feature points
    kp_q, des_q = detector.detectAndCompute(img_query, None)
    kp_t, des_t = detector.detectAndCompute(img_train, None)

    # Select available feature points
    try:
        x = np.array([kp_t[0].pt])
        for i in range(len(kp_t)):
            x = np.append(x, [kp_t[i].pt], axis=0)
        x = x[1:len(x)]

        bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)
        mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
        mean_shift.fit(x)

        labels = mean_shift.labels_
        cluster_centers = mean_shift.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        s = [None] * n_clusters_
        for i in range(n_clusters_):
            l = mean_shift.labels_
            d, = np.where(l == i)
            s[i] = list(kp_t[xx] for xx in d)

        des_t_ = des_t

        best = {'i': 0, 'match_good': [], 'M': None, 'mask': None}
    except (IndexError, ValueError):
        print("[EXCEPTION] No feature points founded")
        return 0, None, (None, None)

    # Start feature matching
    for i in range(n_clusters_):
        task = TaskTime()
        kp_t = s[i]
        l = mean_shift.labels_
        d, = np.where(l == i)
        des_t = des_t_[d, ]

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        des_q = np.float32(des_q)
        des_t = np.float32(des_t)

        # store all the good matches as per Lowe's ratio test.
        try:
            matches = flann.knnMatch(des_q, des_t, k=2)
        except (cv2.error, TypeError):
            cluster_info = f"Zero matches on cluster {i}"
            break
        else:
            match_good = []
            for m, n in matches:
                if m.distance < Param.LOWE_RATIO * n.distance:
                    match_good.append(m)
            cluster_info = __format_cluster_info(i, len(match_good))

        if len(match_good) >= Param.MIN_MATCH:
            src_pts = np.float32([kp_q[m.queryIdx].pt for m in match_good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_t[m.trainIdx].pt for m in match_good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
            if M is not None:
                if len(best['match_good']) <= len(match_good):
                    best['i'] = i
                    best['match_good'] = match_good
                    best['M'] = M
                    best['mask'] = mask

                cluster_info = "Estimated cluster " + cluster_info
            else:
                cluster_info = "No homography on cluster " + cluster_info
        else:
            cluster_info = "Not enough matches on cluster " + cluster_info

        task.change_task_name(cluster_info)
        task.display_time()

    # Creating homography with best matches
    try:
        h, w = img_query.shape[:2]
        edges = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        edges = cv2.perspectiveTransform(edges, best['M'])
        if not __validate_edges(edges):
            raise cv2.error
    except cv2.error:
        print(">>> There is no good homography best cluster")
        return 0, None, (None, None)
    else:
        lines = cv2.polylines(img_train, [np.int32(edges)], True, Param.COLOR_LINE, Param.LINE_WIDTH, cv2.LINE_AA)

        match_mask = best['mask'].ravel().tolist()
        best_params = dict(matchColor=Param.COLOR_MATCH, singlePointColor=None, matchesMask=match_mask, flags=2)

        img_match = cv2.drawMatches(img_query, kp_q, lines, s[best['i']], best['match_good'], None, **best_params)
        cluster_info = __format_cluster_info(best['i'], len(best['match_good']))

        print(f">>> Best match is cluster {cluster_info} ...")
        return len(best['match_good']), img_match, (best['M'], img_train.shape[:2])
