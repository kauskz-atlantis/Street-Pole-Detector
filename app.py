import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread
import json
from pole_detector import PoleModel

class Worker(QObject):
    progress = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, model, image_path):
        super().__init__()
        self.model = model
        self.image_path = image_path

    def run(self):
        for event in self.model.pipeline_stream(self.image_path):
            self.progress.emit(event)
        self.finished.emit()

class PoleDetectionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Utility Pole Detection")
        self.setGeometry(300,150,1000,700)

        self.image_path = None
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_load = QPushButton("Load image")
        self.btn_run = QPushButton("Run Model")
        self.btn_run.setEnabled(False)

        self.status = QLabel("Status: Idle")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_clear = QPushButton("Clear")
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_load)
        button_layout.addWidget(self.btn_run)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_label, stretch=3)
        main_layout.addWidget(self.status)
        main_layout.addWidget(self.btn_clear, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(self.text_output, stretch=1)
        self.setLayout(main_layout)

        self.btn_load.clicked.connect(self.load_img)
        self.btn_run.clicked.connect(self.run_model_thread)
        self.btn_clear.clicked.connect(lambda: self.text_output.setText(""))

        self.model = PoleModel()

    def load_img(self):
        path,_ = QFileDialog.getOpenFileName(self, "Choose an image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            pixmap = QPixmap(path)
            self.image_label.setPixmap(
                pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
            )
            self.text_output.append(f"[INFO] Loaded image: {path}")
            self.status.setText("Status: Idle")
            self.btn_run.setEnabled(True)

    def run_model_thread(self):
        if not self.image_path: return
        self.status.setText("Status: Running...")
        self.text_output.append("[INFO] Running model...")
        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)

        self.thread = QThread()
        self.worker = Worker(self.model, self.image_path)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_model_progress)
        self.worker.finished.connect(self.model_done)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_model_progress(self, event):
        stage = event.get("stage")
        self.show_result(event.get("image"))
        if stage == "result":
            self.text_output.append("[RESULT]: "+json.dumps(event.get("info"), indent=None))
        
    def show_result(self, image):
        h,w,ch = image.shape
        bpl = ch*w
        q_img = QImage(image.data, w, h, bpl, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        )
        
    def model_done(self):
        self.text_output.append("[INFO] Model finished.\n")
        self.status.setText("Status: Finished")
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PoleDetectionGUI()
    gui.show()
    sys.exit(app.exec())