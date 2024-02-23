from PyQt5.QtWidgets import QWidget, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap
from numpy.typing import NDArray
import numpy as np
from qt_widgets import NDarray_to_QPixmap, LabeledDoubleSpinBox, LabeledSliderDoubleSpinBox
from image_tools import im2single, im2uint8

# TODO: this probably belongs in image tools
class ImageControl(QWidget):
    '''
    Contains an image and controls the histogram
    Note that some operations are non-linear (e.g clipping or gamma),
    which means the order in which operations are applied matters.
    The order is the order of the widgets top->down.
    '''

    # TODO be able to reorganize widgets vertically to change the order of operations

    def __init__(self, image: NDArray, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.image = im2single(image)
        self.image_transformed = self.image.copy() 
        self.create_components()
        self.layout_components()

    def create_components(self):

        ## image -------------------------------------------------
        self.image_label = QLabel(self)
        self.image_label.setPixmap(NDarray_to_QPixmap(im2uint8(self.image_transformed)))

        ## controls ----------------------------------------------

        # contrast
        self.contrast = LabeledSliderDoubleSpinBox(self)
        self.contrast.setText('contrast')
        self.contrast.setRange(0.01,10)
        self.contrast.setValue(1.0)
        self.contrast.editingFinished.connect(self.update_histogram)

        # brightness
        self.brightness = LabeledSliderDoubleSpinBox(self)
        self.brightness.setText('brightness')
        self.brightness.setRange(-1,1)
        self.brightness.setValue(0.0)
        self.brightness.editingFinished.connect(self.update_histogram)

        # gamma
        self.gamma = LabeledSliderDoubleSpinBox(self)
        self.gamma.setText('gamma')
        self.gamma.setRange(0,10)
        self.gamma.setValue(1.0)
        self.gamma.editingFinished.connect(self.update_histogram)

        # min
        self.min = LabeledSliderDoubleSpinBox(self)
        self.min.setText('min')
        self.min.setRange(0,1)
        self.min.setValue(0.0)
        self.min.editingFinished.connect(self.update_histogram)

        # max
        self.max = LabeledSliderDoubleSpinBox(self)
        self.max.setText('max')
        self.max.setRange(0,1)
        self.max.setValue(1.0)
        self.max.editingFinished.connect(self.update_histogram)

        ## curve: total transformation applied to pixel values -------

        ## histogram -------------------------------------------------

        ## auto: make the histogram flat 
        self.auto = QPushButton(self)
        self.auto.setText('Auto')
        self.auto.clicked.connect(self.auto_scale)

        ## reset: back to original histogram
        self.reset = QPushButton(self)
        self.reset.setText('Reset')
        self.reset.clicked.connect(self.reset_transform)

    def layout_components(self):

        layout_buttons = QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.auto)
        layout_buttons.addWidget(self.reset)
        layout_buttons.addStretch()

        layout_main = QVBoxLayout(self)
        layout_main.addStretch()
        layout_main.addWidget(self.image_label)
        layout_main.addWidget(self.min)
        layout_main.addWidget(self.max)
        layout_main.addWidget(self.gamma)
        layout_main.addWidget(self.contrast)
        layout_main.addWidget(self.brightness)
        layout_main.addLayout(layout_buttons)
        layout_main.addStretch()

    def update_histogram(self):
        
        # apply transform
        c = self.contrast.value()
        b = self.brightness.value()
        g = self.gamma.value()
        m = self.min.value()
        M = self.max.value()

        self.image_transformed = np.clip(c*(np.clip(self.image, m, M)**g)+b, 0 ,1)

        # update image
        self.image_label.setPixmap(NDarray_to_QPixmap(im2uint8(self.image_transformed)))
        
        # update histogram

    def auto_scale(self):

        m = np.percentile(self.image, 5)
        M = np.percentile(self.image, 99)
        self.min.setValue(m)
        self.max.setValue(M)
        self.update_histogram()
        self.image_label.setPixmap(NDarray_to_QPixmap(im2uint8(self.image_transformed)))
    
    def reset_transform(self):

        # reset parameters
        self.contrast.setValue(1.0)
        self.brightness.setValue(0.0)
        self.gamma.setValue(1.0)
        self.min.setValue(0.0)
        self.max.setValue(1.0)

        # reset image
        self.image_transformed = self.image.copy()
        self.image_label.setPixmap(NDarray_to_QPixmap(im2uint8(self.image_transformed)))

class Landmarks2D(QWidget):
    def __init__(self, fixed: NDArray, moving: NDArray, *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)

        self.moving = moving
        self.fixed = fixed
        self.create_components()
        self.layout_components()

    def create_components(self):

        ## images

        self.moving_label = QLabel(self)
        self.fixed_label = QLabel(self)
        self.overlay = QLabel(self)
    

        ## 2D affine transform parameters

        self.scale_x = LabeledDoubleSpinBox(self)
        self.scale_x.setText('scale x')
        self.scale_x.setRange(-1000,1000)
        self.scale_x.setValue(1)
        self.scale_x.editingFinished.connect(self.update_transfromation)

        self.scale_y = LabeledDoubleSpinBox(self)
        self.scale_y.setText('scale x')
        self.scale_y.setRange(-1000,1000)
        self.scale_y.setValue(1)
        self.scale_y.editingFinished.connect(self.update_transfromation)

        self.shear_x = LabeledDoubleSpinBox(self)
        self.shear_x.setText('scale x')
        self.shear_x.setRange(-1000,1000)
        self.shear_x.setValue(1)
        self.shear_x.editingFinished.connect(self.update_transfromation)

        self.shear_y = LabeledDoubleSpinBox(self)
        self.shear_y.setText('scale x')
        self.shear_y.setRange(-1000,1000)
        self.shear_y.setValue(1)
        self.shear_y.editingFinished.connect(self.update_transfromation)

        self.rotation = LabeledDoubleSpinBox(self)
        self.rotation.setText('scale x')
        self.rotation.setRange(-1000,1000)
        self.rotation.setValue(1)
        self.rotation.editingFinished.connect(self.update_transfromation)

        self.translate_x = LabeledDoubleSpinBox(self)
        self.translate_x.setText('scale x')
        self.translate_x.setRange(-1000,1000)
        self.translate_x.setValue(1)
        self.translate_x.editingFinished.connect(self.update_transfromation)

        self.translate_y = LabeledDoubleSpinBox(self)
        self.translate_y.setText('scale x')
        self.translate_y.setRange(-1000,1000)
        self.translate_y.setValue(1)
        self.translate_y.editingFinished.connect(self.update_transfromation)

        self.transformation_groupbox = QGroupBox('Parameters:')

    def layout_components(self):
        
        layout_params = QVBoxLayout(self.transformation_groupbox)
        layout_params.addStretch()
        layout_params.addWidget(self.scale_x)
        layout_params.addWidget(self.scale_y)
        layout_params.addWidget(self.shear_x)
        layout_params.addWidget(self.shear_y)
        layout_params.addWidget(self.rotation)
        layout_params.addWidget(self.translate_x)
        layout_params.addWidget(self.translate_y)
        layout_params.addStretch()

                
        self.tabs = QTabWidget()
        self.tabs.addTab()

    def update_transfromation(self):
        pass