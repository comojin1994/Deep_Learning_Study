import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import helpers

from tensorflow.keras.backend import set_learning_phase
from tensorflow.keras.models import load_model


def annotate_image(file_path, coordinates):
    """
    Annotates supplied image from predicted coordinates.

    Args:
        file_path: path
            System path of image to annotate
        coordinates: list
            Predicted body part coordinates for image
    """

    # Load raw image
    from PIL import Image, ImageDraw
    image = Image.open(file_path)
    image_width, image_height = image.size
    image_side = image_width if image_width >= image_height else image_height

    # Annotate image
    image_draw = ImageDraw.Draw(image)
    image_coordinates = coordinates[0]
    image = helpers.display_body_parts(image, image_draw, image_coordinates, image_height=image_height,
                                       image_width=image_width, marker_radius=int(image_side / 150))
    image = helpers.display_segments(image, image_draw, image_coordinates, image_height=image_height,
                                     image_width=image_width, segment_width=int(image_side / 100))

    # Save annotated image
    image.save(os.path.normpath(file_path.split('.')[0] + '_tracked.png'))

# (RT, I, II, III or IV)
model_variant = 'RT'

set_learning_phase(0)
model = load_model(os.path.join('models', 'keras', 'EfficientPose{0}.h5'.format(model_variant.upper())),
                   custom_objects={'BilinearWeights': helpers.keras_BilinearWeights,
                                   'Swish': helpers.Swish(helpers.eswish), 'eswish': helpers.eswish,
                                   'swish1': helpers.swish1})

# file_path = f'./utils/MPII.jpg'
file_path = f'./utils/golf.jpeg'
img = cv2.imread(file_path)
h, w = img.shape[0], img.shape[1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
img /= 255.
img = img[np.newaxis, ...]

output = model.predict(img)[-1]
# output = output[0]
# output = np.sum(output, axis=-1)
# output = output[..., np.newaxis]
# output = np.repeat(output, 3, axis=-1)

coord = [helpers.extract_coordinates(output[0,...], h, w)]
annotate_image(file_path, coord)

# plt.imshow(output, cmap='hot')
# plt.show()
