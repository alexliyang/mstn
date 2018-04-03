import tensorflow as tf
import cv2

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def opencv_read(filename, label):
    image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
    return image_decoded, label


# TODO complete the pre process
def parse_function(image_decoded, label):
    # image_string = tf.read_file(filename)
    # image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
    # # image_resized = tf.image.resize_images(image_decoded, [400, 400])
    image_decoded.set_shape([None, None, None])
    image_resized = tf.image.resize_images(image_decoded, [28, 28])

    return image_decoded, label
