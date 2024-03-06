from PyQt5.QtWidgets import (
    QApplication, QWidget, QGraphicsScene, QGraphicsView, 
    QGraphicsTextItem, QGraphicsEllipseItem, QGraphicsItemGroup, 
    QGraphicsItem,  QVBoxLayout, QHBoxLayout,  QPushButton, 
    QCheckBox, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, 
    QFrame)
from PyQt5.QtCore import Qt, QRectF, QPoint, QPointF
from PyQt5.QtGui import QBrush, QPen, QFont, QPixmap
import sys
import cv2
import numpy as np
from qt_widgets import NDarray_to_QPixmap, LabeledDoubleSpinBox, LabeledSliderDoubleSpinBox, LabeledSliderSpinBox
from typing import Protocol
from image_tools import im2single, im2uint8, im2rgb
import pyqtgraph as pg

class ControlPoint(QGraphicsView):

    ZOOM_FACTOR = 0.1 
    POINT_RADIUS = 1.5
    LABEL_OFFSET = 5 
    
    def __init__(self, image: np.ndarray, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.labels = {}
        self.scene = QGraphicsScene()
        self.pixmap_item = self.scene.addPixmap(QPixmap())
        self.setScene(self.scene)
        self.brush = QBrush(Qt.red)
        self.pen = QPen(Qt.red)
        self.font = QFont("Arial", 20)
        self.set_image(image)

    def set_image(self, image: np.ndarray):

        self.image = im2rgb(im2uint8(image))
        self.pixmap_item.setPixmap(NDarray_to_QPixmap(self.image))

    def get_image(self) -> np.ndarray:
        
        return self.image

    def closest_group(self, pos: QPointF):

        # get all group objects
        groups = [
            item 
            for item in self.scene.items() 
            if isinstance(item, QGraphicsItemGroup)
        ]

        # compute the manhattan distance from pos to all group objects
        distances = [
            (item.sceneBoundingRect().center() - pos).manhattanLength() 
            for item in self.scene.items() 
            if isinstance(item, QGraphicsItemGroup)
        ]

        # return the closest group
        if groups:
            return min(zip(groups,distances), key=lambda x: x[1])[0]

    @property    
    def control_points(self):

        # get the center position of all ellipses in the scene
        centers = [
            item.sceneBoundingRect().center() 
            for item in self.scene.items() 
            if isinstance(item, QGraphicsEllipseItem)
        ]
        return centers
    
    def wheelEvent(self, event):
        """
        zoom with wheel
        """
        
        delta = event.angleDelta().y()
        zoom = delta and delta // abs(delta)
        if zoom > 0:
            self.scale(1+self.ZOOM_FACTOR, 1+self.ZOOM_FACTOR)
        else:
            self.scale(1-self.ZOOM_FACTOR, 1-self.ZOOM_FACTOR)

    def mousePressEvent(self, event):
        """
        shift + left-click to add a new control point
        right-click to remove closest control point
        double-click and drag to move control point  
        """
        
        widget_pos = event.pos()
        scene_pos = self.mapToScene(widget_pos)

        if event.modifiers() == Qt.ShiftModifier:
            
            if event.button() == Qt.LeftButton:
            
                # get num
                num = 0 if not self.labels else max(self.labels.values()) + 1

                # add dot
                bbox = QRectF(
                    scene_pos.x() - self.POINT_RADIUS, 
                    scene_pos.y() - self.POINT_RADIUS, 
                    2*self.POINT_RADIUS, 
                    2*self.POINT_RADIUS
                )
                dot = QGraphicsEllipseItem(bbox)
                dot.setBrush(self.brush)
                dot.setPen(self.pen)
                self.scene.addItem(dot)

                # add label
                text_pos = scene_pos + QPoint(self.LABEL_OFFSET,-self.LABEL_OFFSET)
                label = QGraphicsTextItem(str(num))
                label.setPos(text_pos)
                label.setFont(self.font)
                label.setDefaultTextColor(Qt.red)
                self.scene.addItem(label)

                # group dot and label together
                group = self.scene.createItemGroup([dot, label])
                group.setFlags(QGraphicsItem.ItemIsMovable) 
                self.labels[group] = num

        if event.button() == Qt.RightButton:

            # get closest group and delete it and its children
            group = self.closest_group(scene_pos)  
            if group:
                self.labels.pop(group)
                for item in group.childItems():
                    group.removeFromGroup(item)
                    self.scene.removeItem(item)
                self.scene.destroyItemGroup(group)


class ImageWidget(Protocol):
    
    def set_image(self, image: np.ndarray) -> None:
        ...

    def get_image(self) -> np.ndarray:
        ...


class Enhance(QWidget):

    def __init__(self, image_widget: ImageWidget, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.image_widget = image_widget
        self.image = self.image_widget.get_image().copy()
        
        self.num_channels = 3
        self.image = im2single(self.image)
        self.image_enhanced = self.image.copy()

        self.state = {
            'contrast': [1.0 for i in range(self.num_channels)],
            'brightness': [0.0 for i in range(self.num_channels)],
            'gamma': [1.0 for i in range(self.num_channels)],
            'min': [0.0 for i in range(self.num_channels)],
            'max': [1.0 for i in range(self.num_channels)]
        }

        self.create_components()
        self.layout_components()

    def create_components(self):

        # expert mode
        self.expert = QCheckBox(self)
        self.expert.setText('expert mode')
        self.expert.stateChanged.connect(self.expert_mode)

        # channel: which image channel to act on
        self.channel = LabeledSliderSpinBox(self)
        self.channel.setText('channel')
        self.channel.setRange(0,self.num_channels-1)
        self.channel.setValue(0)
        self.channel.valueChanged.connect(self.change_channel)

        # contrast
        self.contrast = LabeledSliderDoubleSpinBox(self)
        self.contrast.setText('contrast')
        self.contrast.setRange(0,10)
        self.contrast.setValue(1.0)
        self.contrast.setSingleStep(0.05)
        self.contrast.valueChanged.connect(self.change_contrast)

        # brightness
        self.brightness = LabeledSliderDoubleSpinBox(self)
        self.brightness.setText('brightness')
        self.brightness.setRange(-1,1)
        self.brightness.setValue(0.0)
        self.brightness.setSingleStep(0.05)
        self.brightness.valueChanged.connect(self.change_brightness)

        # gamma
        self.gamma = LabeledSliderDoubleSpinBox(self)
        self.gamma.setText('gamma')
        self.gamma.setRange(0,10)
        self.gamma.setValue(1.0)
        self.gamma.setSingleStep(0.05)
        self.gamma.valueChanged.connect(self.change_gamma)

        # min
        self.min = LabeledSliderDoubleSpinBox(self)
        self.min.setText('min')
        self.min.setRange(0,1)
        self.min.setValue(0.0)
        self.min.setSingleStep(0.05)
        self.min.valueChanged.connect(self.change_min)

        # max
        self.max = LabeledSliderDoubleSpinBox(self)
        self.max.setText('max')
        self.max.setRange(0,1)
        self.max.setValue(1.0)
        self.max.setSingleStep(0.05)
        self.max.valueChanged.connect(self.change_max)

        ## histogram and curve: total transformation applied to pixel values -------
        self.curve = pg.plot()
        self.curve.setFixedHeight(100)
        self.curve.setYRange(0,1)
        self.histogram = pg.plot()
        self.histogram.setFixedHeight(150)

        ## auto: make the histogram flat 
        self.auto = QPushButton(self)
        self.auto.setText('Auto')
        self.auto.clicked.connect(self.auto_scale)

        ## reset: back to original histogram
        self.reset = QPushButton(self)
        self.reset.setText('Reset')
        self.reset.clicked.connect(self.reset_transform)

        self.curve.hide()
        self.histogram.hide()

    def layout_components(self):

        layout_buttons = QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.auto)
        layout_buttons.addWidget(self.reset)
        layout_buttons.addStretch()

        layout_main = QVBoxLayout(self)
        layout_main.addWidget(self.image_widget)
        layout_main.addWidget(self.expert)
        layout_main.addWidget(self.channel)
        layout_main.addWidget(self.min)
        layout_main.addWidget(self.max)
        layout_main.addWidget(self.gamma)
        layout_main.addWidget(self.contrast)
        layout_main.addWidget(self.brightness)
        layout_main.addWidget(self.curve)
        layout_main.addWidget(self.histogram)
        layout_main.addLayout(layout_buttons)

    def change_channel(self):

        # restore channel state 
        w = self.channel.value()
        self.contrast.setValue(self.state['contrast'][w])
        self.brightness.setValue(self.state['brightness'][w])
        self.gamma.setValue(self.state['gamma'][w])
        self.min.setValue(self.state['min'][w])
        self.max.setValue(self.state['max'][w])

        self.update_histogram()

    def change_brightness(self):
        self.update_histogram()

    def change_contrast(self):
        self.update_histogram()

    def change_gamma(self):
        self.update_histogram()

    def change_min(self):

        w = self.channel.value()
        m = self.min.value() 
        M = self.max.value()

        # if min >= max restore old value 
        if m >= M:
            self.min.setValue(self.state['min'][w])
            
        self.update_histogram()

    def change_max(self):

        w = self.channel.value()
        m = self.min.value() 
        M = self.max.value()

        # if min >= max restore old value 
        if m >= M:
            self.max.setValue(self.state['max'][w])
    
        self.update_histogram()

    def update_histogram(self):
    
        # get parameters
        w = self.channel.value()
        c = self.contrast.value()
        b = self.brightness.value()
        g = self.gamma.value()
        m = self.min.value()
        M = self.max.value()

        # update parameter state 
        self.state['contrast'][w] = c
        self.state['brightness'][w] = b
        self.state['gamma'][w] = g
        self.state['min'][w] = m
        self.state['max'][w] = M

        self.curve.clear()
        self.histogram.clear()

        for ch in range(self.num_channels):

            # transfrom image channel
            I = self.image[:,:,ch].copy()

            I = np.piecewise(
                I, 
                [I<self.state['min'][ch], (I>=self.state['min'][ch]) & (I<=self.state['max'][ch]), I>self.state['max'][ch]],
                [0, lambda x: (x-self.state['min'][ch])/(self.state['max'][ch]-self.state['min'][ch]), 1]
            )
            
            I = np.clip(self.state['contrast'][ch] * (I** self.state['gamma'][ch] -0.5) + self.state['brightness'][ch] + 0.5, 0 ,1)

            self.image_enhanced[:,:,ch] = I

            if self.expert.isChecked():
                
                # update curves
                x = np.arange(0,1,0.02)
                u = np.piecewise(
                    x, 
                    [x<self.state['min'][ch], (x>=self.state['min'][ch]) & (x<=self.state['max'][ch]), x>self.state['max'][ch]],
                    [0, lambda x: (x-self.state['min'][ch])/(self.state['max'][ch]-self.state['min'][ch]), 1]
                )
                y = np.clip(self.state['contrast'][ch] * (u** self.state['gamma'][ch] -0.5) + self.state['brightness'][ch] + 0.5, 0 ,1)
                self.curve.plot(x,y,pen=(ch,3))

                # update histogram
                y, x = np.histogram(I.ravel(), x)
                self.histogram.plot(x,y,stepMode="center", pen=(ch,3))

        # update image
        self.image_widget.set_image(im2uint8(self.image_enhanced))

    def auto_scale(self):

        m = np.percentile(self.image, 5)
        M = np.percentile(self.image, 99)
        self.min.setValue(m)
        self.max.setValue(M)
        self.update_histogram()
        self.image_widget.set_image(im2uint8(self.image_enhanced))
    
    def reset_transform(self):
        
        # reset state
        self.state = {
            'contrast': [1.0 for i in range(self.num_channels)],
            'brightness': [0.0 for i in range(self.num_channels)],
            'gamma': [1.0 for i in range(self.num_channels)],
            'min': [0.0 for i in range(self.num_channels)],
            'max': [1.0 for i in range(self.num_channels)]
        }
                
        # reset parameters
        self.contrast.setValue(1.0)
        self.brightness.setValue(0.0)
        self.gamma.setValue(1.0)
        self.min.setValue(0.0)
        self.max.setValue(1.0)

        # reset image
        self.image_enhanced = self.image.copy()
        self.image_widget.set_image(im2uint8(self.image_enhanced))
        self.update_histogram()

    def expert_mode(self):

        if self.expert.isChecked():
            self.curve.show()
            self.histogram.show()
        else:
            self.curve.hide()
            self.histogram.hide()

        self.update_histogram()

if __name__ == "__main__":

    image = cv2.imread('toy_data/image_00.jpg')[:,:,0]

    app = QApplication(sys.argv)
    cp = ControlPoint(image)
    window = Enhance(cp)
    window.show()
    app.exec()