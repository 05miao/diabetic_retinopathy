import gin
import tensorflow as tf


# Use a mask to crop the black edges in the image
@tf.function
def crop_image_from_gray(img, tol=7):
    # inspired by https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy
    # but here aims to process tensors rather than numpy arrays
    # convert the original RGB image to grayscale
    img = tf.cast(img, tf.float32)
    gray_img = 0.2989*img[:, :, 0]+0.5870*img[:, :, 1]+0.1140*img[:, :, 2]
    # Per thresholding of the grayscale image, non-black pixels will be identified as True, black pixels False.
    # The False column: the uninformative black edges are columns which contain all False elements;
    # the True column: the informative parts are columns which contain at least one True element.
    # Use tf.math.reduce_any, which performs logic OR along the axis (here axis=0, column),
    # to distinguish between these two kinds of columns.
    mask = tf.math.reduce_any(gray_img > tol, 0)
    # Use tf.boolean_mask to drop the False columns in the original RGB image, i.e. crop the black edges.
    # Notice: here axis should be 1
    img = tf.boolean_mask(img, mask, axis=1)
    return img


@gin.configurable
@tf.function
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""

    image = crop_image_from_gray(image, tol=10)
    # Make the cropped rectangle image square
    # Use the length of the rectangle as the side length of the square. Notice: should use tf.shape to get the shape
    side_length = tf.shape(image)[1]
    image = tf.image.resize_with_crop_or_pad(image, side_length, side_length)
    # Resize the square image to 256*256
    image = tf.image.resize(image, size=(img_height, img_width))  # can add method='nearest'
    # Normalize image
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


@tf.function
def augment(image, label):
    """Data augmentation"""

    # Flip image left_right
    image = tf.image.random_flip_left_right(image)
    # Flip image up_down
    image = tf.image.random_flip_up_down(image)
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0, 1)

    return image, label
