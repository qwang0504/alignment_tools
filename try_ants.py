import ants
import cv2
import numpy as np

def ANTsTransform_to_matrix(transform):
    dimension = transform.dimension
    params = transform.parameters
    affine = params.reshape((dimension + 1, dimension))
    affine_matrix = np.eye(dimension + 1)
    affine_matrix[:dimension, :dimension] = affine[:dimension, :dimension]
    affine_matrix[:dimension, dimension] = affine[dimension, :]
    return affine_matrix

# Load your two images
image_fixed = ants.from_numpy(cv2.imread('toy_data/image_00.jpg')[:,:,0])
image_moving = ants.from_numpy(cv2.imread('toy_data/image_00.jpg')[:,:,0])

# Perform image registration
registration = ants.registration(fixed=image_fixed, moving=image_moving, type_of_transform='Affine')

# Get the affine transformation matrix
affine_matrix = registration['fwdtransforms'][0]

Tf = ants.read_transform(affine_matrix)
A = ANTsTransform_to_matrix(Tf)