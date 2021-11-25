import socket
import sys
from datetime import datetime
from os import path
from shutil import copy, copytree
import imghdr

import cv2
import qrcode
from PyQt5 import QtMultimedia, uic
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import (QAction, QActionGroup, QApplication, QMainWindow,
                             QMessageBox)

from config_parser import UI, File, Param, Path
from crack_detector import create_model, finetune, predict, set_gpu
from cv2_helper import make_overlay, change_color
from data_manager import split_train_valid
from feature_matcher import apply_homography, make_homography
from file_manager import (clean_tree, get_all_files, open_file_dialog,
                          remove_tree)
from gui_helper import (TaskTime, assign_shortcut, delayed_function,
                        parse_window_size, show_window)

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer


_client_ip = None
_section = 0
_section_max = 1


class __MainWindow(QMainWindow):
    ''' Main crack manager window. '''

    def __init__(self, parent=None):
        super().__init__(parent)

        # Load UI
        self.ui = uic.loadUi(File.UI, self)
        self.setWindowTitle(f"{UI.TITLE} {UI.VERSION}")
        self.setGeometry(*parse_window_size(UI.WINDOW))
        assign_shortcut(self)

        # Apply configs to UI
        self.ui.toggleAutoCapture.setChecked(UI.AUTO_CAPTURE)
        self.ui.toggleAutoMatch.setChecked(UI.AUTO_MATCH)
        self.ui.toggleAutoDetect.setChecked(UI.AUTO_DETECT)
        self.ui.toggleAutoApply.setChecked(UI.AUTO_APPLY)
        self.ui.toggleAutoSend.setChecked(UI.AUTO_SEND)

        # Set class variables
        self.camera = {'id': -1, 'name': "", 'capture': None}
        self.last_used = {'capture_as_crack': 0, 'capture_as_match': 0, 'holo_as_match': 0,
                          'crack_as_result': 0, 'match_as_result': 0}
        self.homography_info = None

        # Initiallize camera
        self.preview_timer = QTimer(self, interval=UI.TIME_UPDATE)
        self.preview_timer.timeout.connect(self.update_image)
        self.register_cameras()
        self.set_camera()

        # Setup segmentation network
        set_gpu()
        self.model = create_model(Param.MODEL, Param.BACKBONE, Param.ACTIVATION)
        self.register_weights()

        # Init section combo
        self.ui.comboSections.clear()
        self.ui.comboSections.addItem('0')
        self.sectionSelected(_section)

    def showAbout(self):
        ''' Show application authos info. '''

        QMessageBox.information(self, "About", UI.ABOUT, QMessageBox.Close)

    # region Camera

    def register_cameras(self):
        ''' Register available camera devices to menu item. '''

        videoDevicesGroup = QActionGroup(self)
        videoDevicesGroup.setExclusive(True)

        for i, deviceName in enumerate(QtMultimedia.QCamera.availableDevices()):
            description = QtMultimedia.QCamera.deviceDescription(deviceName)
            videoDeviceAction = QAction(description, videoDevicesGroup)
            videoDeviceAction.setCheckable(True)
            videoDeviceAction.setData(i)
            if self.camera['id'] == -1:
                self.camera['id'] = i
                self.camera['name'] = description
                videoDeviceAction.setChecked(True)
            self.ui.menuDevices.addAction(videoDeviceAction)

        videoDevicesGroup.triggered.connect(self.set_camera)

    def set_camera(self, action=None):
        ''' Prepare camera for capture. '''

        if action:
            self.camera['id'] = action.data()
            self.camera['name'] = action.text()

        self.togglePreview(True)

    def togglePreview(self, toggle):
        ''' Toggle camera preview. '''

        self.ui.buttonCapture.setEnabled(toggle)
        self.ui.toggleButtonPreview.setChecked(toggle)
        self.toggleButtonPreview.setText("Stop Camera" if toggle else "Start Camera")

        if toggle:
            self.camera['capture'] = cv2.VideoCapture(self.camera['id'])
            self.preview_timer.start()

            print("Camera started: " + self.camera['name'])
        else:
            if self.camera['capture']:
                self.camera['capture'].release()
            self.camera['capture'] = None
            self.preview_timer.stop()

            print("Camera stopped: " + self.camera['name'])

    def update_image(self):
        ''' Called when frame update. '''

        success, img = self.camera['capture'].read()
        if success:
            qformat = QImage.Format_Indexed8
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
            outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
            outImage = outImage.rgbSwapped()

            self.ui.labelPreview.setPixmap(QPixmap.fromImage(outImage))
        else:
            print("Camera update failed: " + self.camera['name'])
            self.togglePreview(False)

    def makeCapture(self):
        ''' Capture image and save. '''

        success, img = self.camera['capture'].read()
        if success:
            cv2.imwrite(self.file_capture, img)
            print("Capture success: " + self.camera['name'])
            self.on_capture()
        else:
            print("Capture failed: " + self.camera['name'])
            self.togglePreview(False)

    # endregion

    # region Network

    def send_mssage(self, message):
        ''' Send message to holo client. '''

        if _client_ip:
            with(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as client:
                print(f"Sending {message} to {_client_ip[0]}:{Param.PORT + 1}")
                client.sendto(message.encode('utf-8'), (_client_ip[0], Param.PORT + 1))

    def format_file(self, name):
        return name.replace('@', str(_section))

    def sectionSelected(self, index):
        ''' Called when combo item selected. '''

        global _section
        _section = index

        self.file_index = self.format_file(File.INDEX)
        self.file_holo = self.format_file(File.HOLO)
        self.file_capture = self.format_file(File.CAPTURE)
        self.file_crack = self.format_file(File.CRACK)
        self.file_match = self.format_file(File.MATCH)
        self.file_result = self.format_file(File.RESULT)
        self.file_overlay = self.format_file(File.OVERLAY)
        self.file_logs = self.format_file(File.TRN_LOGS)
        self.path_capture = self.format_file(Path.CAPTURE)
        self.path_crack = self.format_file(Path.CRACK)
        self.path_crackgt = self.format_file(Path.CRACK_GT)
        self.path_data = self.format_file(Path.DATA)

        self.ui.labelCapture.setPixmap(QPixmap(self.file_capture))
        self.ui.labelHolo.setPixmap(QPixmap(self.file_holo))
        self.ui.labelCrack.setPixmap(QPixmap(self.file_crack))
        self.ui.labelMatch.setPixmap(QPixmap(self.file_match))
        self.ui.labelOverlay.setPixmap(QPixmap(self.file_overlay))

        print(f"Canged section: {_section}")

        self.send_mssage(f'section:{_section}')

    def networkChanged(self):
        ''' Called when network status is changed. '''

        if _client_ip:
            self.ui.buttonDisconnect.setEnabled(True)
            self.ui.textIP.setText(f"{_client_ip[0]}:{_client_ip[1]}")
            self.ui.labelHolo.setPixmap(QPixmap(File.CONNECTED))
        else:
            self.ui.buttonDisconnect.setEnabled(False)
            self.ui.textIP.setText("")
            self.ui.labelHolo.setPixmap(QPixmap(File.QRCODE))
            self.ui.buttonSend.setEnabled(False)

    def disconnect(self):
        ''' Disconnect current client. '''

        global _client_ip
        _client_ip = None
        self.networkChanged()

    def messageReceived(self, message):
        ''' Called when text message received. '''

        if message:
            print("Received: " + message)
            self.on_holo_received()

    def sectionNew(self):
        ''' Add new section and move. '''

        global _section_max
        self.ui.comboSections.addItem(str(_section_max))
        self.ui.comboSections.setCurrentIndex(_section_max)
        _section_max += 1

    def sectionClear(self):
        ''' Clear current section. '''

        self.removeCapture()
        self.removeHolo()
        self.removeCrack()
        self.removeMatch()
        self.removeResult()

        self.send_mssage(f'clear:{_section}')

    def sendResult(self):
        ''' Send result to holo client. '''

        self.send_mssage(f'result:{_section}')

    # endregion

    # region CrackDetect

    def register_weights(self):
        self.ui.comboWeights.clear()
        self.ui.comboWeights.addItem(File.BASE_MODEL)
        try:
            self.ui.comboWeights.addItems(get_all_files(Path.CKPT))
        except FileNotFoundError:
            pass

    def weightsSelected(self, index):
        weights = self.ui.comboWeights.currentText()
        self.model.load_weights(weights)
        print(f">>> Load weights from {weights}")

    def init_train_data(self, path_anns, shape):
        ''' Make initial training data index. '''

        heights, widths = shape
        files_ann = get_all_files(path_anns)
        with open(self.file_index, 'w') as file:
            file.write(f'{heights},{widths}\n')
            for file_ann in files_ann:
                file.write(path.basename(file_ann) + '=TP\n')

    def makeCrack(self, callback=False):
        ''' Make crack image. '''

        if self.ui.buttonDetectCrack.isEnabled():
            updated = self.update_file_time(self.file_capture, 'capture_as_crack')
            if not callback or updated:
                self.send_mssage('busy')
                task = TaskTime("makeCrack")
                img_crack, shape = predict(self.model, self.file_capture, self.path_capture, self.path_crack)
                cv2.imwrite(self.file_crack, img_crack)
                self.init_train_data(self.path_crack, shape)
                task.display_time()
                self.send_mssage('free')

                self.on_crack_created()

    # endregion

    # region FeatureMatch

    def try_make_homography(self, img_query, img_train, detector):
        ''' Try make homography with detector. '''

        success, img_match, self.homography_info = make_homography(
            img_query, img_train, detector, Param.EQUALIZER)
        if success:
            cv2.imwrite(self.file_match, img_match)
            return success

    def makeMatch(self, callback=False):
        ''' Start feature matching. '''

        if self.ui.buttonFeatureMatch.isEnabled():
            updated_capture = self.update_file_time(self.file_capture, 'capture_as_match')
            updated_holo = self.update_file_time(self.file_holo, 'holo_as_match')
            if not callback or updated_capture or updated_holo:
                self.send_mssage('busy')
                task = TaskTime('makeMatch')
                img_query = cv2.imread(self.file_capture)
                img_train = cv2.imread(self.file_holo)

                if self.try_make_homography(img_query, img_train, Param.FEATURE_DETECTOR_1):
                    task.display_time()
                    self.on_feature_matched()
                # elif self.try_make_homography(img_query, img_train, Param.FEATURE_DETECTOR_2):
                #    task.display_time()
                #    self.on_feature_matched()
                self.send_mssage('free')

    def makeResult(self, callback=False):
        ''' Apply homography to crack image. '''

        if self.ui.buttonApplyHomography.isEnabled():
            updated_crack = self.update_file_time(self.file_crack, 'crack_as_result')
            updated_match = self.update_file_time(self.file_match, 'match_as_result')
            if not callback or updated_crack or updated_match:
                self.send_mssage('busy')
                task = TaskTime('makeResult')
                img_holo = cv2.imread(self.file_holo)
                img_crack = cv2.imread(self.file_crack)

                img_result = apply_homography(img_crack, self.homography_info)
                img_overlay = make_overlay(img_holo, img_result)

                cv2.imwrite(self.file_overlay, img_overlay)
                cv2.imwrite(self.file_result, img_result)
                task.display_time()
                self.send_mssage('free')

                self.on_homography_applied()

    # endregion

    # region FileOperations

    def file_saved(self, file):
        ''' Show message when file saved. '''

        print(">>> File saved: " + file)

    def update_file_time(self, file, keyword):
        ''' Update last used file time to prevent auto executing functions. '''

        current = path.getmtime(file)
        if self.last_used[keyword] != current:
            self.last_used[keyword] = current
            return True

    def openCapture(self):
        ''' Open captured image from disk. '''

        src = open_file_dialog("Open Wall Image", Path.SAMPLES, f'Image Files {Param.IMAGE_EXTENSIONS}')
        if src:
            print(">>> Opening file: " + src)
            if path.exists(copy(src, self.file_capture)):
                self.on_capture()

    def removeCapture(self):
        remove_tree(self.file_capture)
        self.ui.labelCapture.clear()
        self.ui.buttonRemoveCapture.setEnabled(False)
        self.ui.buttonFeatureMatch.setEnabled(False)
        self.ui.buttonDetectCrack.setEnabled(False)

    def removeHolo(self):
        remove_tree(self.file_holo)

        self.ui.labelHolo.setPixmap(QPixmap(File.QRCODE))
        self.ui.buttonFeatureMatch.setEnabled(False)

    def removeCrack(self):
        remove_tree(self.file_crack)
        self.ui.labelCrack.clear()
        self.ui.buttonRemoveCrack.setEnabled(False)
        self.ui.buttonApplyHomography.setEnabled(False)
        self.last_used['capture_as_crack'] = 0

    def removeMatch(self):
        remove_tree(self.file_match)
        self.ui.labelMatch.clear()
        self.ui.buttonRemoveMatch.setEnabled(False)
        self.ui.buttonApplyHomography.setEnabled(False)
        self.last_used['capture_as_match'] = 0
        self.last_used['holo_as_match'] = 0
        self.homography_info = None

    def removeResult(self):
        remove_tree(self.file_result)
        self.ui.labelOverlay.clear()
        self.ui.buttonRemoveResult.setEnabled(False)
        self.ui.buttonSend.setEnabled(False)
        self.last_used['crack_as_result'] = 0
        self.last_used['match_as_result'] = 0

    def archiveAll(self):
        save_path = Path.ARCHIVES + '/' + datetime.now().strftime("%Y-%m-%d_%p-%I-%M-%S")
        clean_tree([save_path])
        copytree(Path.RESULTS, save_path)
        self.file_saved(save_path)

    def cleanAll(self):
        reply = QMessageBox.question(self, "Clean", "Are you sure want to delete all caches?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.removeCapture()
            self.removeHolo()
            self.removeCrack()
            self.removeMatch()
            self.removeResult()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Exit", "Save results to archive folder?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if reply == QMessageBox.Save:
            self.archiveAll()
            remove_tree(Path.RESULTS)
            event.accept()
        elif reply == QMessageBox.Discard:
            remove_tree(Path.RESULTS)
            event.accept()
        else:
            event.ignore()

    # endregion

    # region Callbacks

    def on_capture(self):
        ''' Invoke when capture. '''

        self.file_saved(self.file_capture)
        self.ui.labelCapture.setPixmap(QPixmap(self.file_capture))
        self.ui.buttonRemoveCapture.setEnabled(True)
        self.ui.buttonDetectCrack.setEnabled(True)
        self.ui.buttonFeatureMatch.setEnabled(path.exists(self.file_holo))

        self.removeCrack()
        self.removeMatch()
        self.removeResult()

        delayed_function(lambda: self.makeMatch(True), self.ui.toggleAutoMatch.isChecked(), 10)
        delayed_function(lambda: self.makeCrack(True), self.ui.toggleAutoDetect.isChecked(), 20)

    def on_holo_received(self):
        ''' Invoke when file received. '''

        self.file_saved(self.file_holo)
        self.ui.labelHolo.setPixmap(QPixmap(self.file_holo))
        self.ui.buttonFeatureMatch.setEnabled(path.exists(self.file_capture))

        self.removeMatch()
        self.removeResult()

        delayed_function(lambda: self.makeMatch(True), self.ui.toggleAutoMatch.isChecked(), 10)
        delayed_function(lambda: self.makeCapture(), self.ui.toggleAutoCapture.isChecked(), 20)

    def on_crack_created(self):
        ''' Invoke when capture. '''

        self.file_saved(self.file_crack)
        self.ui.labelCrack.setPixmap(QPixmap(self.file_crack))
        self.ui.buttonRemoveCrack.setEnabled(True)
        self.ui.buttonApplyHomography.setEnabled(path.exists(self.file_match))

        self.removeResult()

        delayed_function(lambda: self.makeResult(True), self.ui.toggleAutoApply.isChecked(), 10)
        delayed_function(lambda: self.makeMatch(True), self.ui.toggleAutoMatch.isChecked(), 20)

    def on_feature_matched(self):
        ''' Invoke when feature match is success. '''

        self.file_saved(self.file_match)
        self.ui.labelMatch.setPixmap(QPixmap(self.file_match))
        self.ui.buttonRemoveMatch.setEnabled(True)
        self.ui.buttonApplyHomography.setEnabled(path.exists(self.file_crack))

        self.removeResult()

        delayed_function(lambda: self.makeResult(True), self.ui.toggleAutoApply.isChecked(), 10)
        delayed_function(lambda: self.makeCrack(True), self.ui.toggleAutoDetect.isChecked(), 20)

    def on_homography_applied(self):
        ''' Invoke when homography is applied. '''

        self.file_saved(self.file_result)
        self.file_saved(self.file_overlay)
        self.ui.labelOverlay.setPixmap(QPixmap(self.file_overlay))
        self.ui.buttonRemoveResult.setEnabled(True)
        self.ui.buttonSend.setEnabled(True)

        delayed_function(self.sendResult, self.ui.toggleAutoSend.isChecked(), 10)

    # endregion


class ServerThread(QThread):
    signal_network_changed = pyqtSignal()
    signal_message_received = pyqtSignal(str)

    def __init__(self):
        QThread.__init__(self)

    def listen(self, server_ip, server_port):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.bind((server_ip, server_port))

        addr = f"{server_ip}:{server_port}:{path.basename(File.HOLO)}:{path.basename(File.RESULT)}"
        print(">>> Server started: " + addr)
        qrcode.make(addr).save(File.QRCODE)
        self.signal_network_changed.emit()

    def run(self):
        while True:
            global _client_ip
            bytesAddressPair = self.server.recvfrom(1024)
            if _client_ip:
                self.signal_message_received.emit(bytesAddressPair[0].decode('utf-8'))
            else:
                _client_ip = bytesAddressPair[1]
                print(f">>> Client connected: {_client_ip[0]}:{_client_ip[1]}")
                self.signal_network_changed.emit()

    def close(self):
        self.server.close()
        self.server = None


class FTPThread(QThread):
    def __init__(self):
        QThread.__init__(self)
        self.server = None

    def listen(self, server_ip, server_port):
        authorizer = DummyAuthorizer()
        authorizer.add_anonymous(Path.RESULTS, perm="elradfmw")

        handler = FTPHandler
        handler.authorizer = authorizer

        self.server = FTPServer((server_ip, server_port), handler)

    def run(self):
        self.server.serve_forever()


def __main():
    ''' CrackManager initializer. '''

    clean_tree([Path.RESULTS])

    app = QApplication(sys.argv)
    mainWindow = __MainWindow()

    server_ip = socket.gethostbyname(socket.gethostname())

    serverThread = ServerThread()
    serverThread.signal_network_changed.connect(mainWindow.networkChanged)
    serverThread.signal_message_received.connect(mainWindow.messageReceived)
    serverThread.listen(server_ip, Param.PORT)
    serverThread.start()

    show_window(mainWindow, not UI.FULLSCREEN)
    print(">>> Starting application")

    ftpThread = FTPThread()
    ftpThread.listen(server_ip, 21)
    ftpThread.start()

    sys.exit(app.exec_())


if __name__ == '__main__':
    __main()
