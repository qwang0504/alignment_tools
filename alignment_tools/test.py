from alignment_tools.align_affine_2D import AlignAffine2D
from PyQt5.QtWidgets import QApplication
import sys
import cv2

FIXED = 'toy_data/image_00.jpg'
MOVING = 'toy_data/image_00.jpg'

app = QApplication(sys.argv)
window = AlignAffine2D(cv2.imread(FIXED), cv2.imread(MOVING))
window.show()
app.exec()
