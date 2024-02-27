import ants
import cv2

# Load your two images
image_fixed = ants.from_numpy(cv2.imread('toy_data/image_00.jpg')[:,:,0])
image_moving = ants.from_numpy(cv2.imread('toy_data/image_00.jpg')[:,:,0])

# Perform image registration
registration = ants.registration(fixed=image_fixed, moving=image_moving, type_of_transform='Affine')

# Get the affine transformation matrix
affine_matrix = registration['fwdtransforms'][0]

# Print the affine matrix
print("Affine Transformation Matrix:")
print(affine_matrix)