from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsTextItem, QGraphicsEllipseItem, QGraphicsItemGroup, QGraphicsItem
from PyQt5.QtCore import Qt, QRectF, QPoint, QPointF
from PyQt5.QtGui import QBrush, QPen, QFont, QTransform
import sys
import cv2
import numpy as np
from qt_widgets import NDarray_to_QPixmap

class ControlPoint(QGraphicsView):

    ZOOM_FACTOR = 0.1 
    POINT_RADIUS = 1.5
    LABEL_OFFSET = 5 
    
    def __init__(self, image: np.ndarray, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.image = image
        self.labels = {}

        self.scene = QGraphicsScene()
        self.pixmap_item = self.scene.addPixmap(NDarray_to_QPixmap(image))
        self.setScene(self.scene)
        self.brush = QBrush(Qt.red)
        self.pen = QPen(Qt.red)
        self.font = QFont("Arial", 20)

    def set_image(self, image: np.ndarray):
        
        self.image = image
        self.pixmap_item.setPixmap(NDarray_to_QPixmap(image))

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
                

if __name__ == "__main__":

    image = cv2.imread('toy_data/image_00.jpg')[:,:,0]

    app = QApplication(sys.argv)
    window = ControlPoint(image)
    window.show()
    app.exec()