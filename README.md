# alignment_tools

Landmark based and automatic alignment tools

# Installation

```
git clone https://github.com/ElTinmar/alignment_tools.git
conda env create -f alignment_tools.yml
```

# How to use 

```
from alignment_tools.align_affine_2D import AlignAffine2D
from PyQt5.QtWidgets import QApplication
import sys
import cv2

FIXED = 'toy_data/image_00.jpg'
MOVING = 'toy_data/image_02.jpg'

app = QApplication(sys.argv)
window = AlignAffine2D(cv2.imread(FIXED)[:,:,0], cv2.imread(MOVING)[:,:,0])
window.show()
app.exec()
```


