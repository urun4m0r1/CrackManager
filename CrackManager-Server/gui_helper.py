import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut

from config_parser import UI, Param


class TaskTime():
    ''' Print task time. '''

    def __init__(self, task_name="", display_init=False):
        self.task_name = task_name
        self.__time_start = time.time()
        if display_init:
            print(f"{self.task_name} started")

    def change_task_name(self, task_name): self.task_name = task_name

    def display_time(self):
        self.time_spent = round(time.time() - self.__time_start, Param.DIGITS)
        print(f"{self.task_name} spent {self.time_spent} seconds")
        self.__time_start = time.time()
        return self.time_spent

    def get_time(self): return self.time_spent


def delayed_function(function, condition=True, frame_delay=0):
    ''' Execute function after delay if toggle is checked. '''

    if condition:
        QTimer.singleShot(UI.TIME_UPDATE * frame_delay, function)


def parse_window_size(string):
    ''' Return window size and offset from string. '''

    window_x = int(string.split('+')[0].split('x')[0])
    window_y = int(string.split('+')[0].split('x')[1])
    offset_x = int(string.split('+')[1])
    offset_y = int(string.split('+')[2])
    return offset_x, offset_y, window_x, window_y


def show_window(window, flag):
    ''' Shortcut for toggle fullscreen. '''

    (window.showNormal if flag else window.showFullScreen)()


def assign_shortcut(window):
    ''' Assign shortcuts. '''

    QShortcut(QKeySequence(Qt.Key_Escape), window).activated.connect(window.close)
    QShortcut(QKeySequence(Qt.ALT + Qt.Key_Return),
              window).activated.connect(lambda: show_window(window, window.isFullScreen()))
