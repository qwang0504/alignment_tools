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
np_image = cv2.imread('toy_data/image_00.jpg')[:,:,0]
ants_image = ants.from_numpy(np_image)

image_fixed = ants.from_numpy(np.transpose(cv2.imread('toy_data/image_00.jpg')[:,:,0]))
image_moving = ants.from_numpy(np.transpose(cv2.imread('toy_data/image_01.jpg')[:,:,0]))

image_fixed.plot()

# Perform image registration
registration = ants.registration(fixed=image_fixed, moving=image_moving, type_of_transform='Similarity', verbose = True)

# Get the affine transformation matrix
Tf = ants.read_transform(registration['fwdtransforms'][0])
Tr = ants.read_transform(registration['invtransforms'][0])

Af = ANTsTransform_to_matrix(Tf)
Ar = ANTsTransform_to_matrix(Tr)

np.allclose(Af,Ar) # fwdtransforms and invtransforms are the same, you need to be careful and potentially inverse the matrix before use

#
f = ants.utils.get_pointer_string(image_fixed)
m = ants.utils.get_pointer_string(image_moving)
initx = ["[%s,%s,1]" % (f, m)]

Tinv = ants.invert_ants_transform(Tf)

#

Tf = ants.create_ants_transform(dimension=2,matrix=np.array([[1.4, 0.9],[0.9,1.1]]),translation=(10.0,4.12))
ANTsTransform_to_matrix(Tf)