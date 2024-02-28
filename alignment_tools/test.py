from alignment_tools.align_affine_2D import AlignAffine2D
from PyQt5.QtWidgets import QApplication
import sys
import cv2

FIXED = cv2.imread('toy_data/image_00.jpg')[:,:,0]
MOVING = cv2.imread('toy_data/image_02.jpg')[:,:,0]

app = QApplication(sys.argv)
window = AlignAffine2D(FIXED, MOVING)
window.show()
app.exec()
