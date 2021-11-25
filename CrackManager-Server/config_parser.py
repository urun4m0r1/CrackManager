import re
import sys
from configparser import ConfigParser

_config = ConfigParser()
_config.read('resources/settings.ini')
try:
    _ui = _config['UI']
    _param = _config['Param']
    _path = _config['Path']
    _file = _config['File']
except KeyError:
    print("File 'resources/settings.ini' or section [UI], [Param], [Path], [File] doesnt exist")
    sys.exit()


def _parse(section, key, val_type=str, val_range=None):
    ''' Parse key with value type in range. '''

    try:
        if val_type is str:
            val = section[key]
            if val_range and not re.compile(val_range).match(val):
                raise AssertionError
            else:
                return val
        elif val_type in (int, float):
            val = val_type(section[key])
            if val_range and not min(val_range) <= val <= max(val_range):
                raise AssertionError
            else:
                return val
        elif val_type is bool:
            true_perfix = ('1', 'y', 't', 'yes', 'true')
            false_perfix = ('0', 'n', 'f', 'no', 'false')
            val = section[key].lower()
            if val in true_perfix:
                return True
            elif val in false_perfix:
                return False
            else:
                val_range = (true_perfix, false_perfix)
                raise AssertionError
        else:
            raise TypeError
    except KeyError:
        print(f"Key '{key}' dosent exist")
        sys.exit()
    except TypeError:
        print(f"Type '{val_type}' is not callable")
        sys.exit()
    except ValueError:
        print(f"Key '{key} = {val}' is not an {val_type}")
        sys.exit()
    except AssertionError:
        print(f"Key '{key} = {val}' is out of range {val_range}")
        sys.exit()


def _parse_color(key):
    ''' Parse color with validayion. '''

    regex_color = r'^(?:(?:[A-F0-9]{2}){3,4}|[A-F0-9]{3})$'
    code = _parse(_param, key, val_range=regex_color)
    return tuple(int(code[i:i+2], 16) for i in (0, 2, 4))


def _append_path(parent, key, need_index=False):
    ''' Append path with validayion. '''

    if need_index:
        regex_path = r'^([^<>\|\?\*\"\\\/]+)[^<>:\|\?\*\"\\]*\_@/$'
    else:
        regex_path = r'^([^<>\|\?\*\"\\\/]+)[^<>:\|\?\*\"\\]*\/$'
    return parent + _parse(_path, key, val_range=regex_path)


def _append_file(parent, key, need_index=False):
    ''' Append file path with validayion. '''

    if need_index:
        regex_file = r'^([^<>:\|\?\*\"\\\/]*)_@\.?([^<>:\|\?\*\"\\\/\.]+)$'
    else:
        regex_file = r'^([^<>:\|\?\*\"\\\/]*)\.?([^<>:\|\?\*\"\\\/\.]+)$'
    return parent + _parse(_file, key, val_range=regex_file)


class UI:
    ''' Configs for UIs. '''

    # General
    TITLE = _parse(_ui, 'TITLE')
    VERSION = _parse(_ui, 'VERSION')
    ABOUT = _parse(_ui, 'ABOUT')
    WINDOW = _parse(_ui, 'WINDOW', val_range=r'^[1-9]\d+x[1-9]\d+\+\d+\+\d+$')
    FULLSCREEN = _parse(_ui, 'FULLSCREEN', bool)
    POPUP_FULLSCREEN = _parse(_ui, 'POPUP_FULLSCREEN', bool)
    TIME_UPDATE = round(1000 / _parse(_ui, 'FPS', int, (10, 240)))
    # Initial Settings
    AUTO_CAPTURE = _parse(_ui, 'AUTO_CAPTURE', bool)
    AUTO_DETECT = _parse(_ui, 'AUTO_DETECT', bool)
    AUTO_MATCH = _parse(_ui, 'AUTO_MATCH', bool)
    AUTO_APPLY = _parse(_ui, 'AUTO_APPLY', bool)
    AUTO_SEND = _parse(_ui, 'AUTO_SEND', bool)


class Param:
    ''' Configs for parameters. '''

    # General
    DIGITS = _parse(_param, 'DIGITS', int, (1, 10))
    PORT = _parse(_param, 'PORT', int, (1024, 65535))
    IMAGE_EXTENSIONS = _parse(_param, 'IMAGE_EXTENSIONS', val_range=r'^\((?:\*\.\S+\s)+\*\.\S+\)$')
    # Feature Matcher
    FEATURE_DETECTOR_1 = _parse(_param, 'FEATURE_DETECTOR_1', val_range=r'akaze|brisk_fast|brisk_slow|orb_fast|orb_slow')
    FEATURE_DETECTOR_2 = _parse(_param, 'FEATURE_DETECTOR_2', val_range=r'akaze|brisk_fast|brisk_slow|orb_fast|orb_slow')
    EQUALIZER = _parse(_param, 'EQUALIZER', val_range=r'equalize|match|none')
    LOWE_RATIO = _parse(_param, 'LOWE_RATIO', float, (0, 1))
    MIN_MATCH = _parse(_param, 'MIN_MATCH', int, (0, sys.maxsize))
    BRISK_THRESH_SLOW = _parse(_param, 'BRISK_THRESH_SLOW', int, (0, sys.maxsize))
    BRISK_THRESH_FAST = _parse(_param, 'BRISK_THRESH_FAST', int, (BRISK_THRESH_SLOW, sys.maxsize))
    ORB_FEATURES_FAST = _parse(_param, 'ORB_FEATURES_FAST', int, (0, sys.maxsize))
    ORB_FEATURES_SLOW = _parse(_param, 'ORB_FEATURES_SLOW', int, (ORB_FEATURES_FAST, sys.maxsize))
    # Results display
    EDGE = _parse(_param, 'EDGE', int, (0, 100))
    EDGE_CUT = _parse(_param, 'EDGE_CUT', int, (0, 100))
    DILATE = _parse(_param, 'DILATE', int, (0, sys.maxsize))
    LINE_WIDTH = _parse(_param, 'LINE_WIDTH', int, (0, sys.maxsize))
    COLOR_LINE = _parse_color('COLOR_LINE')
    COLOR_MATCH = _parse_color('COLOR_MATCH')
    COLOR_CRACK = _parse_color('COLOR_CRACK')
    TRANSPARENCY = _parse(_param, 'TRANSPARENCY', int, (0, 255))
    # Crack Detect
    TRN_VLD_RATIO = [float(x)*0.1 for x in _parse(_param, 'TRN_VLD_RATIO',
                                                      val_range=r'[0-9.]+:[0-9.]+').split(':')]
    TRN_VLD_RATIO_RE_TRAIN = [float(x)*0.1 for x in _parse(_param, 'TRN_VLD_RATIO_RE_TRAIN',
                                                      val_range=r'[0-9.]+:[0-9.]+').split(':')]
    MODEL = _parse(_param, 'MODEL', val_range=r'unet|fpn|linknet|pspnet')
    BACKBONE = _parse(_param, 'BACKBONE')
    ACTIVATION = _parse(_param, 'ACTIVATION',
                        val_range=r'elu|softmax|selu|softplus|softsign|relu|tanh|sigmoid|hard_sigmoid|exponential|linear')
    SCALE = _parse(_param, 'SCALE', float, (0, 10))
    BLOCK = int(_parse(_param, 'BLOCK', val_range=r'32|64|96|128|160|192|224|256|288|320'))
    LEARNING_RATE = _parse(_param, 'LEARNING_RATE', float, (0, 1))
    LEARNING_RATE_RE_TRAIN = _parse(_param, 'LEARNING_RATE_RE_TRAIN', float, (0, 1))
    BATCH_SIZE = _parse(_param, 'BATCH_SIZE', int, (1, sys.maxsize))
    EPOCHS = _parse(_param, 'EPOCHS', int, (0, sys.maxsize))
    EPOCHS_SAVEPOINT = _parse(_param, 'EPOCHS_SAVEPOINT', int, (0, EPOCHS))
    EPOCHS_RE_TRAIN = _parse(_param, 'EPOCHS_RE_TRAIN', int, (0, sys.maxsize))
    EPOCHS_SAVEPOINT_RE_TRAIN = _parse(_param, 'EPOCHS_SAVEPOINT_RE_TRAIN', int, (0, EPOCHS_RE_TRAIN))


class Path:
    ''' Configs for file paths. '''

    # General
    ROOT = _append_path('', 'ROOT')
    PERFIX_IMG = _append_path('', 'PERFIX_IMG')
    PERFIX_ANN = _append_path('', 'PERFIX_ANN')
    PERFIX_TRN = _append_path('', 'PERFIX_TRN')
    PERFIX_VLD = _append_path('', 'PERFIX_VLD')
    TRN_IMG = PERFIX_TRN + PERFIX_IMG
    TRN_ANN = PERFIX_TRN + PERFIX_ANN
    VLD_IMG = PERFIX_VLD + PERFIX_IMG
    VLD_ANN = PERFIX_VLD + PERFIX_ANN
    # Root subfolders
    SAMPLES = _append_path(ROOT, 'SAMPLES')
    ARCHIVES = _append_path(ROOT, 'ARCHIVES')
    RESOURCES = _append_path(ROOT, 'RESOURCES')
    CKPT = _append_path(ROOT, 'CKPT')
    RESULTS = _append_path(ROOT, 'RESULTS')
    # Results subfolders
    DATA = _append_path(RESULTS, 'DATA', True)
    CAPTURE = _append_path(RESULTS, 'CAPTURE', True)
    CRACK = _append_path(RESULTS, 'CRACK', True)
    CRACK_GT = _append_path(RESULTS, 'CRACK_GT', True)


class File:
    ''' Configs for file paths. '''

    # Resources subfiles
    UI = _append_file(Path.RESOURCES, 'UI')
    BASE_MODEL = _append_file(Path.RESOURCES, 'BASE_MODEL')
    CONNECTED = _append_file(Path.RESOURCES, 'CONNECTED')
    # Results subfiles
    QRCODE = _append_file(Path.RESULTS, 'QRCODE')
    INDEX = _append_file(Path.RESULTS, 'INDEX', True)
    HOLO = _append_file(Path.RESULTS, 'HOLO', True)
    CAPTURE = _append_file(Path.RESULTS, 'CAPTURE', True)
    CRACK = _append_file(Path.RESULTS, 'CRACK', True)
    MATCH = _append_file(Path.RESULTS, 'MATCH', True)
    RESULT = _append_file(Path.RESULTS, 'RESULT', True)
    OVERLAY = _append_file(Path.RESULTS, 'OVERLAY', True)
    TRN_LOGS = _append_file(Path.RESULTS, 'TRN_LOGS', True)
