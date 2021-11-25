from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel

from config_parser import UI
from gui_helper import assign_shortcut, parse_window_size, show_window


class QImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None

    def setPixmap(self, pixmap, callback=False):
        super().setPixmap(pixmap)
        if not callback:
            self.pixmap = pixmap
            self.resizeEvent(None)

    def clear(self):
        super().clear()
        self.pixmap = None

    def resizeEvent(self, event):
        if self.pixmap:
            self.setPixmap(self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation), True)


class QImagePopup(QImageLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{UI.TITLE} {UI.VERSION}")
        self.setGeometry(*parse_window_size(UI.WINDOW))
        self.setWindowModality(Qt.ApplicationModal)
        self.setAlignment(Qt.AlignCenter)
        assign_shortcut(self)

    def mousePressEvent(self, event): self.close()


class QImageWidget(QImageLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__popup = QImagePopup()

    def setPixmap(self, pixmap, callback=False):
        super().setPixmap(pixmap, callback)
        self.__popup.setPixmap(pixmap, callback)

    def mousePressEvent(self, event):
        if self.pixmap:
            show_window(self.__popup, not UI.POPUP_FULLSCREEN)
