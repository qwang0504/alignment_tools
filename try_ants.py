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
image_fixed = ants.from_numpy(np.transpose(cv2.imread('toy_data/image_00.jpg')[:,:,0]))
image_moving = ants.from_numpy(np.transpose(cv2.imread('toy_data/image_00.jpg')[:,:,0]))

image_fixed.plot()

# Perform image registration
registration = ants.registration(fixed=image_fixed, moving=image_moving, type_of_transform='Similarity', verbose = True)

# Get the affine transformation matrix
affine_matrix = registration['fwdtransforms'][0]

Tf = ants.read_transform(affine_matrix)
Tr = ants.read_transform(registration['invtransforms'][0])

A = ANTsTransform_to_matrix(Tf)


# 
f = ants.utils.get_pointer_string(image_fixed)
m = ants.utils.get_pointer_string(image_moving)
initx = ["[%s,%s,1]" % (f, m)]

Tinv = ants.invert_ants_transform(Tf)

#

Tf = ants.create_ants_transform(dimension=2,matrix=np.array([[1.4, 0.9],[0.9,1.1]]),translation=(10.0,4.12))
ANTsTransform_to_matrix(Tf)