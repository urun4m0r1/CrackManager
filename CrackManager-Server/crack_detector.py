import cv2
import keras
from keras import backend as K
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

from config_parser import Param, Path
from cv2_helper import assemble_image, slice_image, change_color
from data_albumentator import get_preprocess, get_trn_augment, get_vld_augment
from data_manager import Dataloder, Dataset
from file_manager import clean_tree, get_all_files
from gui_helper import TaskTime


def set_gpu():
    ''' Initialize GPU memory. '''

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError:
            print("Memory growth must be set before GPUs have been initialized")


def create_model(architecture, backbone, activation):
    ''' Initialize traning model. '''

    # define optomizer
    optim = keras.optimizers.Adam(Param.LEARNING_RATE)
    total_loss = sm.losses.binary_focal_dice_loss
    metrics = [sm.metrics.IOUScore(threshold=0.5),
               sm.metrics.FScore(threshold=0.5),
               'accuracy']

    # compile keras model with defined optimozer, loss and metrics
    if architecture == 'unet':
        model = sm.Unet(backbone, classes=4, activation=activation)
    elif architecture == 'linknet':
        model = sm.Linknet(backbone, classes=4, activation=activation)
    elif architecture == 'fpn':
        model = sm.FPN(backbone, classes=4, activation=activation)
    elif architecture == 'pspnet':
        model = sm.PSPNet(backbone, classes=4, activation=activation)
    else:
        print("[EXCEPTION] Wrong architecture name")

    model.compile(optim, total_loss, metrics)
    return model


def finetune(model, path_data, save_path, logs_path, retrain=False):
    ''' Finetune model from dataset and save weights to path. '''

    train_dataloader, valid_dataloader = generate_dataloader_set(path_data, retrain)

    # define callbacks for learning rate scheduling and best checkpoints saving
    model_path = save_path + 'model_{epoch:03d}_{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(filepath=model_path,
                                       monitor='val_loss', mode='min',
                                       period=Param.EPOCHS_SAVEPOINT_RE_TRAIN if retrain else Param.EPOCHS_SAVEPOINT,
                                       save_weights_only=True, save_best_only=not retrain,
                                       verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min',
                                  factor=0.05, patience=5, min_lr=1e-6,
                                  verbose=1)
    csv_logger = CSVLogger(logs_path)

    if retrain:
        K.set_value(model.optimizer.learning_rate, Param.LEARNING_RATE_RE_TRAIN)

    # train model
    history = model.fit_generator(
        generator=train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=Param.EPOCHS_RE_TRAIN if retrain else Param.EPOCHS,
        callbacks=[model_checkpoint, reduce_lr, csv_logger],
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )

    print(">>> Complete training process")
    return model_path, history


def generate_dataloader_set(path_data, retrain=False):
    train_dataloader = generate_dataloader(path_data + Path.TRN_IMG, path_data + Path.TRN_ANN, not retrain)
    valid_dataloader = generate_dataloader(path_data + Path.VLD_IMG, path_data + Path.VLD_ANN, False)

    return train_dataloader, valid_dataloader


def generate_dataloader(path_x, path_y, is_train):
    ''' gGnerate dataloader from paths. '''

    if is_train:
        dataset = Dataset(path_x, path_y, 'train', False)
        dataloder = Dataloder(dataset, batch_size=Param.BATCH_SIZE, shuffle=True)
        assert dataloder[0][0].shape == (Param.BATCH_SIZE, Param.BLOCK, Param.BLOCK, 3)
        assert dataloder[0][1].shape == (Param.BATCH_SIZE, Param.BLOCK, Param.BLOCK, 4)
    else:
        dataset = Dataset(path_x, path_y, 'valid', False)
        dataloder = Dataloder(dataset, batch_size=1, shuffle=False)
        assert dataloder[0][0].shape == (1, Param.BLOCK, Param.BLOCK, 3)
        assert dataloder[0][1].shape == (1, Param.BLOCK, Param.BLOCK, 4)

    return dataloder


def evaluate(model, dataloader, epoch=0, logs=None):
    ''' Evaluate model from dataset. '''

    scores = model.evaluate_generator(dataloader)
    metrics = [sm.metrics.IOUScore(threshold=0.5),
               sm.metrics.FScore(threshold=0.5),
               'accuracy']

    log_text = " - test_loss: {:.5}".format(scores[0])
    for metric, value in zip(metrics, scores[1:]):
        log_text += " - test_{}: {:.5}".format(metric, value)
    print(log_text)


def predict(model, path_img, x_path, y_path):
    ''' Predict from dataset with model and save results to path. '''

    shape = slice_image(path_img, '{:04}.png', Param.BLOCK, Param.SCALE)
    clean_tree([y_path])
    _, _, num = shape;

    for i, img_slice in enumerate(get_all_files(x_path)):
        task = TaskTime(f"makeCrack:{i}/{num}")
        img = cv2.imread(img_slice)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        img_predict = model.predict(img).round()
        img_predict = np.squeeze(img_predict)
        h, w = img_predict.shape[:2]
        img_crack = np.zeros((h, w, 3))
        img_crack[..., :3] = img_predict[..., :3] * 255
        img_crack = change_color(img_crack, (255, 0, 0), (255, 255, 255))
        cv2.imwrite(img_slice.replace(x_path, y_path), img_crack)
        task.display_time()

    img_assembles, shape = assemble_image(y_path, shape, Param.SCALE)
    h, w = cv2.imread(path_img).shape[:2]
    h_, w_ = img_assembles.shape[:2]
    top = bottom = abs(int((h - h_) / 2))
    left = right = abs(int((w - w_) / 2))
    img_assembles = cv2.copyMakeBorder(img_assembles, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return img_assembles, shape
