from landmark_2D import Landmarks2D
from PyQt5.QtWidgets import QApplication
import sys

FIXED = 'toy_data/image_00.png'
MOVING = 'toy_data/image_01.png'

app = QApplication(sys.argv)
window = Landmarks2D(FIXED, MOVING)
window.show()
app.exec()
