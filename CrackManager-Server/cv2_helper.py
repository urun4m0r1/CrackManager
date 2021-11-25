import cv2
import numpy as np
from skimage.exposure import cumulative_distribution
from os import path

from file_manager import get_all_files, clean_tree


def get_all_images(path_img):
    ''' Get all images to array in path. '''
    return [cv2.imread(file) for file in get_all_files(path_img)]


def imresize(img, width=None, height=None, keep_aspect=True):
    ''' Resizes a image automatically. '''

    h, w = img.shape[:2]

    if height and width:
        dim = (width, height)
        interpolation = cv2.INTER_CUBIC if height > h or width > w else cv2.INTER_AREA
    elif height and not width:
        new_width = int(w * (height / h)) if keep_aspect else width
        dim = (new_width, height)
        interpolation = cv2.INTER_CUBIC if height > h else cv2.INTER_AREA
    elif not height and width:
        new_height = int(h * (width / w)) if keep_aspect else height
        dim = (width, new_height)
        interpolation = cv2.INTER_CUBIC if width > w else cv2.INTER_AREA
    else:
        return img

    img = cv2.resize(img, dim, interpolation=interpolation)
    return img


def match_resize(img_a, img_b, match_small=True, match_exact=False):
    ''' Match two images size keeping same aspect ratio. '''
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    if (w_a > w_b and match_small) or (w_a < w_b and not match_small):
        if match_exact:
            img_a = imresize(img_a, width=w_b, height=h_b)
        else:
            img_a = imresize(img_a, width=w_b)
    elif (w_a > w_b and not match_small) or (w_a < w_b and match_small):
        if match_exact:
            img_b = imresize(img_b, width=w_a, height=h_a)
        else:
            img_b = imresize(img_b, width=w_a)
    return img_a, img_b


def __compute_cdf(img):
    ''' Computes the CDF of an image im as 2D numpy ndarray. '''

    img_cdf, bin_centers = cumulative_distribution(img)

    # pad the beginning and ending pixels and their CDF values
    img_cdf = np.insert(img_cdf, 0, [0] * bin_centers[0])
    img_cdf = np.append(img_cdf, [1] * (255 - bin_centers[-1]))
    return img_cdf


def match_histogram(img_source, img_template):
    ''' Find closest pixel-matches corresponding to the CDF of the input image. '''

    cdf_source = __compute_cdf(img_source)
    cdf_template = __compute_cdf(img_template)
    pixels = np.interp(cdf_source, cdf_template, np.arange(256))

    img_matched = (np.reshape(pixels[img_source.ravel()], img_source.shape)).astype(np.uint8)
    return img_matched, img_template


def make_overlay(img_background, img_alpha):
    ''' Add overlay to image. '''
    img_background, img_alpha = match_resize(img_background, img_alpha)
    h, w, c = img_alpha.shape[:3]

    img_result = np.zeros((h, w, 3), np.uint8)
    alpha = img_alpha[:, :, 3] / 255
    img_result[:, :, 0] = (1 - alpha) * img_background[:, :, 0] + alpha * img_alpha[:, :, 0]
    img_result[:, :, 1] = (1 - alpha) * img_background[:, :, 1] + alpha * img_alpha[:, :, 1]
    img_result[:, :, 2] = (1 - alpha) * img_background[:, :, 2] + alpha * img_alpha[:, :, 2]
    return img_result


def change_color(img, src, dst):
    ''' Change color from src to dst. '''
    r1, g1, b1 = src
    r2, g2, b2 = dst

    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    img[:, :, :3][mask] = [r2, g2, b2]
    return img
    

def fill_edge(img, size, level):
    ''' Fill image edges with level. '''

    h, w = img.shape[:2]
    if size > 0:
        img[:, 0: size] = level
        img[0: size, :] = level
        img[:, w - size: w] = level
        img[h - size: h, :] = level
    return img


def dliatie_crack(img, size):
    ''' Enlarge crack by dliating. '''

    kernel = np.ones((size, size), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def __imrescale(img, scale):
    ''' Automactally choose interpolation to resize image. '''

    if scale > 1:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif scale < 1:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        return img


def __crop_image(img, block_size, scale=1):
    ''' Crop image with multiple of desired block size. '''

    img = __imrescale(img, scale)
    height, width = img.shape[:2]

    # Resize if smaller than block size
    minsize = min(height, width)
    if minsize < block_size:
        if minsize == height:
            img = imresize(img, height=block_size)
        else:
            img = imresize(img, width=block_size)

    # Crop image with block size
    height, width = img.shape[:2]
    new_width = width - (width % block_size)
    new_height = height - (height % block_size)

    x = int((width - new_width) / 2)
    y = int((height - new_height) / 2)
    w = int((width + new_width) / 2)
    h = int((height + new_height) / 2)
    return img[y:h, x:w]


def slice_image(path_img, name, block, scale):
    ''' slice image to multiple block. '''

    path_slice = path.splitext(path_img)[0] + '/'
    clean_tree([path_slice])

    img = cv2.imread(path_img)
    img = __crop_image(img, block, scale)

    h, w = img.shape[:2]
    heights = h // block
    widths = w // block
    num = 0
    for i in range(heights):
        for j in range(widths):
            img_sliced = img[i * block: (i + 1) * block, j * block: (j + 1) * block]
            cv2.imwrite(path_slice + name.format(num), img_sliced)
            num += 1
    return heights, widths, num


def assemble_image(path_slice, shape, scale):
    ''' slice image to multiple block. '''

    heights, widths, num = shape
    imgs_sliced = get_all_images(path_slice)

    imgs_h = []
    for i in range(heights):
        imgs_w = []
        for j in range(widths):
            imgs_w.append(imgs_sliced[i * widths + j])
        img_sliced_h = np.hstack(imgs_w)
        imgs_h.append(img_sliced_h)
    img_assembles = np.vstack(imgs_h)
    img_assembles = __imrescale(img_assembles, 1/scale)
    return img_assembles, (heights, widths)
