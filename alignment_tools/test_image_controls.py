from alignment_tools.landmark_2D import ImageControl
from PyQt5.QtWidgets import QApplication
import sys

IMAGE = 'toy_data/image_00.jpg'

app = QApplication(sys.argv)
window = ImageControl(IMAGE)
window.show()
app.exec()
