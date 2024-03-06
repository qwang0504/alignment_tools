from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QTableWidget, QTableWidgetItem, QHeaderView, 
    QFrame, QGroupBox)
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from qt_widgets import LabeledDoubleSpinBox
from image_tools import im2single, im2uint8, im2gray, ControlPoint, Enhance, ImageViewer
import ants


def affine_transformation_matrix_to_params(A: np.ndarray):
    '''
    Decompose affine transformation into shear, rotation, scaling and translation
    This decomposition assumes the following order: scaling, shear, rotation and 
    translation.

        | 1  0  t_x |   | cos(theta) -sin(theta) 0 |   | 1 hx  0 |   | s_x  0  0 |
    A = | 0  1  t_y | X | sin(theta)  cos(theta) 0 | X | 0  1  0 | X |  0  s_y 0 |
        | 0  0   1  |   |     0            0     1 |   | 0  0  1 |   |  0   0  1 |
    '''
    
    RHS = A[:2,:2]
    s_x = np.sqrt(RHS[0,0]**2 + RHS[1,0]**2)
    s_y = np.linalg.det(RHS)/s_x
    theta = np.arctan2(RHS[1,0],RHS[0,0])
    h_x = (RHS[:,0] @ RHS[:,1])/np.linalg.det(RHS)
    t_x = A[0,2]
    t_y = A[1,2]

    return (s_x, s_y, theta, h_x, t_x, t_y)


def ANTsTransform_to_matrix(transform) -> np.ndarray:
    # transform ANTsTransform object into a numpy affine transformation matrix
    # ANTsTransform.fixed_parameters stores the center of rotation
    # ANTsTransform.parameters store additional rotation/scale/shear/translation
    
    dimension = transform.dimension
    params = transform.parameters
    rotation_center = transform.fixed_parameters
    affine = params.reshape((dimension + 1, dimension))

    RHS = np.eye(dimension + 1)
    T = np.eye(dimension + 1)
    C = np.eye(dimension + 1)
    RHS[:dimension, :dimension] = affine[:dimension, :dimension]
    T[:dimension, dimension] = affine[dimension, :]
    C[:dimension, dimension] = rotation_center

    affine_matrix = T @ C @ RHS @ np.linalg.inv(C) 

    return affine_matrix


class AlignAffine2D(QWidget):

    def __init__(self, fixed: np.ndarray, moving: np.ndarray, *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)

        self.moving = im2uint8(im2gray(im2single(moving)))
        self.fixed = im2uint8(im2gray(im2single(fixed)))
        self.moving_transformed = self.moving
        self.overlay = np.dstack((self.fixed,np.zeros_like(self.fixed),np.zeros_like(self.fixed)))
        self.affine_transform = np.eye(3,dtype=float)
        self.create_components()
        self.layout_components()
        self.setWindowTitle("Registration2D")

    def create_components(self):

        ## images

        self.moving_cp = ControlPoint(self.moving)
        self.moving_label = Enhance(self.moving_cp)
        self.fixed_cp = ControlPoint(self.fixed)
        self.fixed_label = Enhance(self.fixed_cp)
        self.overlay_viewer = ImageViewer(self.overlay)
        self.overlay_label = Enhance(self.overlay_viewer)

        ## 2D affine transform parameters

        self.scale_x = LabeledDoubleSpinBox(self)
        self.scale_x.setText('scale x')
        self.scale_x.setRange(-1000,1000)
        self.scale_x.setValue(1.0)
        self.scale_x.setSingleStep(0.05)
        self.scale_x.valueChanged.connect(self.callback_spinboxes)

        self.scale_y = LabeledDoubleSpinBox(self)
        self.scale_y.setText('scale y')
        self.scale_y.setRange(-1000,1000)
        self.scale_y.setValue(1.0)
        self.scale_y.setSingleStep(0.05)
        self.scale_y.valueChanged.connect(self.callback_spinboxes)

        self.shear_x = LabeledDoubleSpinBox(self)
        self.shear_x.setText('shear x')
        self.shear_x.setRange(-1000,1000)
        self.shear_x.setValue(0.0)
        self.shear_x.setSingleStep(0.01)
        self.shear_x.valueChanged.connect(self.callback_spinboxes)

        self.shear_y = LabeledDoubleSpinBox(self)
        self.shear_y.setText('shear y')
        self.shear_y.setRange(-1000,1000)
        self.shear_y.setValue(0.0)
        self.shear_y.setSingleStep(0.01)
        self.shear_y.valueChanged.connect(self.callback_spinboxes)

        self.rotation = LabeledDoubleSpinBox(self)
        self.rotation.setText('rotate (deg)')
        self.rotation.setRange(-100_000,100_000)
        self.rotation.setValue(0)
        self.rotation.setSingleStep(0.5)
        self.rotation.valueChanged.connect(self.callback_spinboxes)

        self.translate_x = LabeledDoubleSpinBox(self)
        self.translate_x.setText('translate x')
        self.translate_x.setRange(-100_000,100_000)
        self.translate_x.setValue(0.0)
        self.translate_x.setSingleStep(1.0)
        self.translate_x.valueChanged.connect(self.callback_spinboxes)

        self.translate_y = LabeledDoubleSpinBox(self)
        self.translate_y.setText('translate y')
        self.translate_y.setRange(-100_000,100_000)
        self.translate_y.setValue(0.0)
        self.translate_y.setSingleStep(1.0)
        self.translate_y.valueChanged.connect(self.callback_spinboxes)

        self.transformation_matrix_table = QTableWidget(self)
        self.transformation_matrix_table.setRowCount(3)
        self.transformation_matrix_table.setColumnCount(3)  
        self.transformation_matrix_table.setItem(0,0,QTableWidgetItem('1.0'))
        self.transformation_matrix_table.setItem(0,1,QTableWidgetItem('0.0'))
        self.transformation_matrix_table.setItem(0,2,QTableWidgetItem('0.0'))
        self.transformation_matrix_table.setItem(1,0,QTableWidgetItem('0.0'))
        self.transformation_matrix_table.setItem(1,1,QTableWidgetItem('1.0'))
        self.transformation_matrix_table.setItem(1,2,QTableWidgetItem('0.0'))
        self.transformation_matrix_table.setItem(2,0,QTableWidgetItem('0.0'))
        self.transformation_matrix_table.setItem(2,1,QTableWidgetItem('0.0'))
        self.transformation_matrix_table.setItem(2,2,QTableWidgetItem('1.0'))
        self.transformation_matrix_table.cellChanged.connect(self.callback_table)
        self.transformation_matrix_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.transformation_matrix_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.transformation_matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.transformation_matrix_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.transformation_matrix_table.horizontalHeader().hide()
        self.transformation_matrix_table.verticalHeader().hide()
        self.transformation_matrix_table.setFrameShape(QFrame.NoFrame)
        self.transformation_matrix_table.setMaximumHeight(100)

        self.transformation_groupbox = QGroupBox('Parameters:')
        
        self.reset = QPushButton(self)
        self.reset.setText('Reset transformation')
        self.reset.clicked.connect(self.reset_transform)

        self.align_cp = QPushButton(self)
        self.align_cp.setText('Align with control points')
        self.align_cp.clicked.connect(self.align_control_points)

        self.align_auto = QPushButton(self)
        self.align_auto.setText('Align automatically')
        self.align_auto.clicked.connect(self.align_automatically)

    def layout_components(self):
        
        layout_params = QVBoxLayout(self.transformation_groupbox)
        layout_params.addWidget(self.scale_x)
        layout_params.addWidget(self.scale_y)
        layout_params.addWidget(self.shear_x)
        layout_params.addWidget(self.shear_y)
        layout_params.addWidget(self.rotation)
        layout_params.addWidget(self.translate_x)
        layout_params.addWidget(self.translate_y)
        layout_params.addWidget(self.transformation_matrix_table)
        layout_params.addWidget(self.reset)
        layout_params.addWidget(self.align_cp)
        layout_params.addWidget(self.align_auto)
        layout_params.addStretch()

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.transformation_groupbox)

        # open widgets as separate windwos
        self.fixed_label.show()
        self.fixed_label.setWindowTitle("fixed image")

        self.moving_label.show()
        self.moving_label.setWindowTitle("moving image")

        self.overlay_label.show()
        self.overlay_label.setWindowTitle("overlay")

    def update_images(self):
        # update overlay image

        h, w = self.fixed.shape[:2]
        self.moving_transformed = cv2.warpAffine(self.moving, self.affine_transform[:2,:], (w,h))
        self.overlay = np.dstack((self.fixed,self.moving_transformed,np.zeros_like(self.fixed)))
        self.overlay_label.set_image(self.overlay) 
        self.overlay_label.reset_transform()

    def update_spinboxes(self, s_x, s_y, theta, h_x, t_x, t_y):
        # update values without triggering callbacks

        self.scale_x.blockSignals(True)
        self.scale_x.setValue(s_x)
        self.scale_x.blockSignals(False)

        self.scale_y.blockSignals(True)
        self.scale_y.setValue(s_y)
        self.scale_y.blockSignals(False)

        self.rotation.blockSignals(True)
        self.rotation.setValue(np.rad2deg(theta))
        self.rotation.blockSignals(False)

        self.shear_x.blockSignals(True)
        self.shear_x.setValue(h_x)
        self.shear_x.blockSignals(False)

        self.shear_y.blockSignals(True)
        self.shear_y.setValue(0)
        self.shear_y.blockSignals(False)

        self.translate_x.blockSignals(True)
        self.translate_x.setValue(t_x)
        self.translate_x.blockSignals(False)

        self.translate_y.blockSignals(True)
        self.translate_y.setValue(t_y)
        self.translate_y.blockSignals(False)

    def update_table(self):
        # update values without triggering callbacks

        for i in range(3):
            for j in range(3):
                self.transformation_matrix_table.blockSignals(True)
                self.transformation_matrix_table.setItem(i,j,QTableWidgetItem(f'{self.affine_transform[i,j]:2f}'))
                self.transformation_matrix_table.blockSignals(False)

    def callback_table(self, row: int, col: int):

        it = self.transformation_matrix_table.item(row,col)
        self.affine_transform[row,col] = float(it.text())

        # update parameters
        (s_x, s_y, theta, h_x, t_x, t_y) = affine_transformation_matrix_to_params(self.affine_transform)
        self.update_spinboxes(s_x, s_y, theta, h_x, t_x, t_y)

        self.update_images()

    def callback_spinboxes(self):
        
        sh_x = self.shear_x.value()
        sh_y = self.shear_y.value()
        c = np.cos(np.deg2rad(self.rotation.value()))
        s = np.sin(np.deg2rad(self.rotation.value()))
        t_x = self.translate_x.value()
        t_y = self.translate_y.value()
        sc_x = self.scale_x.value()
        sc_y = self.scale_y.value()

        H = np.array([[ 1.0, sh_x,   0],
                    [sh_y,  1.0,   0],
                    [   0,    0, 1.0]])
        
        R = np.array([[c, -s,   0],
                    [s,  c,   0],
                    [0,  0, 1.0]])

        S = np.array([[sc_x,    0,   0],
                    [0,    sc_y,   0],
                    [0,       0, 1.0]])
        
        T = np.array([[1.0,   0, t_x],
                    [0,   1.0, t_y],
                    [0,     0, 1.0]])

        self.affine_transform = T @ R @ H @ S

        self.update_table()
        self.update_images()

    def align_control_points(self):

        # find affine transformation between two sets of points
        a = [[pos.x(), pos.y(), 1] for pos in self.fixed_cp.control_points]
        b = [[pos.x(), pos.y(), 1] for pos in self.moving_cp.control_points]
        A = np.transpose(np.linalg.lstsq(b, a, rcond=None)[0])
        self.affine_transform = A

        # update GUI
        (s_x, s_y, theta, h_x, t_x, t_y) = affine_transformation_matrix_to_params(self.affine_transform)
        self.update_spinboxes(s_x, s_y, theta, h_x, t_x, t_y)
        self.update_table()
        self.update_images()

    def reset_transform(self):

        self.affine_transform = np.eye(3,dtype=float)

        # update GUI
        (s_x, s_y, theta, h_x, t_x, t_y) = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        self.update_spinboxes(s_x, s_y, theta, h_x, t_x, t_y)
        self.update_table()
        self.update_images()

    def align_automatically(self):

        # Affine registration with ANTs
        # CAUTION It looks like images are flipped by ANTs (need to transpose) 
        registration = ants.registration(
            fixed = ants.from_numpy(np.transpose(self.fixed)), 
            moving = ants.from_numpy(np.transpose(self.moving_transformed)), 
            type_of_transform = 'Affine',
            aff_iterations= (1000, 500, 500, 100),
            aff_sampling=64, 
            aff_random_sampling_rate=1.0,
            aff_shrink_factors=(6, 4, 2, 1), 
            aff_smoothing_sigmas=(3, 2, 1, 0),
            verbose = True
        )

        # Get the affine transformation matrix
        # CAUTION invtransforms and fwdtransforms are the same for affine (https://github.com/ANTsX/ANTsPy/issues/340) 
        # Here I need to take the inverse
        transform_file = registration['fwdtransforms'][0]
        Tf = ants.read_transform(transform_file)
        A = np.linalg.inv(ANTsTransform_to_matrix(Tf))

        # compose with current transformation
        self.affine_transform = A @ self.affine_transform

        # update GUI
        (s_x, s_y, theta, h_x, t_x, t_y) = affine_transformation_matrix_to_params(self.affine_transform)
        self.update_spinboxes(s_x, s_y, theta, h_x, t_x, t_y)
        self.update_table()
        self.update_images()

    def closeEvent(self, event):
        self.moving_label.close()
        self.fixed_label.close()
        self.overlay_label.close()
