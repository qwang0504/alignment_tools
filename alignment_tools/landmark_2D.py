from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap
from numpy.typing import NDArray
import numpy as np
from qt_widgets import NDarray_to_QPixmap

class Landmarks2D(QWidget):
    def __init__(self, fixed: NDArray, moving: NDArray, *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)

        self.moving = moving
        self.fixed = fixed
        self.create_components()
        self.layout_components()

    def create_components(self):
        pass

    def layout_components(self):
        pass