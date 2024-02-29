import ants
import numpy as np
import cv2

def ANTsTransform_to_matrix(transform):
    # transform ANTsTransform object into a numpy affine transformation matrix
    
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

FIXED = ants.image(cv2.imread('toy_data/image_00.jpg')[:,:,0])
MOVING = ants.image(cv2.imread('toy_data/image_04.jpg')[:,:,0])

reg = ants.registration()