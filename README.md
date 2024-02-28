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

FIXED = cv2.imread('toy_data/image_00.jpg')[:,:,0]
MOVING = cv2.imread('toy_data/image_02.jpg')[:,:,0]

app = QApplication(sys.argv)
window = AlignAffine2D(FIXED, MOVING)
window.show()
app.exec()
```

# GUI

Use mouse wheel to zoom, right-click drag towards the edges to pan, left-click to create a control point
and shift-left-click to remove the nearest control point.

# Instructions

Place at least 3 control points on each image before computing the control point-based transformation.   
Control points are numbered, they need to match between the two images.  

Then refine using auto registration (uses ANTs). You may need to click a few times to get it
right.   

Do auto registration only when you're close enough with control points, otherwise it won't converge.

