from os import path

import cv2

from config_parser import File, Param, Path
from crack_detector import create_model, finetune, predict, set_gpu, evaluate, generate_dataloader
from cv2_helper import get_all_images, slice_image
from data_manager import split_train_valid
from file_manager import get_all_files, get_newest_file, clean_tree
from gui_helper import TaskTime


def __train_crack_detector(model):
    ''' Crack detector train code '''

    path_dataset = Path.SAMPLES + 'data/default/'
    split_train_valid(Path.RESOURCES + 'default/' + Path.PERFIX_IMG,
                      Path.RESOURCES + 'default/' + Path.PERFIX_ANN,
                      path_dataset)

    finetune(model, path_dataset, Path.CKPT, Path.SAMPLES + 'training_logs.csv')


def __test_crack_detector(model, mode, cond=0):
    ''' Crack detector test code '''

    paths_wall = get_all_files(Path.SAMPLES + 'wall/')
    paths_crack = [path_wall.replace('wall', 'crack').replace('.jpg', '.png') for path_wall in paths_wall]

    for i, (path_wall, path_crack) in enumerate(zip(paths_wall, paths_crack)):
        task = TaskTime(f"test_crack_detector_{path_crack}")
        slice_image(path_wall, '{:04}.png', Param.BLOCK, Param.SCALE)

        x_path = path_wall.replace('.jpg', '/')
        y_path = path_crack.replace('.png', '/')

        if mode == 'predict':
            img_crack, _ = predict(model, path_wall, x_path, y_path)
            cv2.imwrite(path_crack, img_crack)
            task.display_time()
        else:
            slice_image(path_crack.replace('crack', 'crackgt'), '{:04}.png', Param.BLOCK, Param.SCALE)
            y_path = y_path.replace('crack', 'crackgt')
            if path.exists(y_path):
                if mode == 'train' and i == cond:
                    path_dataset = path_wall.replace('.jpg', '/').replace('wall', 'data')
                    split_train_valid(x_path, y_path, path_dataset, True)
                    finetune(model, path_dataset, Path.CKPT, Path.SAMPLES + f'training_logs_{i+1}.csv', True)
                    task.display_time()
                    __test_crack_detector(__model, 'predict')
                elif mode == 'evaluate':
                    dataloader = generate_dataloader(x_path, y_path, False)
                    evaluate(model, dataloader)
                    task.display_time()
            else:
                print("[EXCEPTION] Crack ground truth not exist!")


if __name__ == '__main__':
    set_gpu()
    __model = create_model(Param.MODEL, Param.BACKBONE, Param.ACTIVATION)

    clean_tree([Path.SAMPLES + 'crack/'])
    clean_tree([Path.SAMPLES + 'data/'])

    try:
        __weights = get_newest_file(Path.CKPT)
        #__weights = File.BASE_MODEL
    except FileNotFoundError:
        __weights = File.BASE_MODEL
    if path.exists(__weights):
        __model.load_weights(__weights)
        print(f">>> Load weights from {__weights}")
        __test_crack_detector(__model, 'predict', 0)
    else:
        print(">>> Train from beginning")
        __train_crack_detector(__model)
