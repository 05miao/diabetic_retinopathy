import tensorflow as tf
import matplotlib.pyplot as plt
import logging


# adapted with modification from https://keras.io/examples/vision/grad_cam/
# https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
# https://www.kaggle.com/nguyenhoa/dog-cat-classifier-gradcam-with-tensorflow-2-0/notebook

# ------------- find convolutional layer name ------------- #
def find_conv_layer(model):
    for layer in reversed(model.layers):
        if 'Conv2D' in type(layer).__name__:  # name of Class
            logging.info(f'The last convolutional layer is {layer.name}')
            return layer.name
    raise ValueError("Couldn't find any convolutional layer in this model")
# ------------- find convolutional layer name ------------- #


# ------------- GradGAM ------------ #
def gradcam(image, model, layer_name=None, index=None):
    """

    Args:
        image: 3-D Tensor or Numpy array. An image with shape (img_height, img_width, channel=3), without batch dim
        model: model after training
        layer_name: String. Name of the last convolutional layer
        index: 'None' or Integer.
            'None' as default, because the predicted class for the image remains to be computed. Model have n
            (c = 0, 1, ..., n-1) outputs (before softmax). The predicted class will be c, if the c-th output is maximal.
            Integer when specified. Specify the c-th output you are interested in. It may not be the maximal output.

    Returns:
        jet_heatmap: Tensor. GradGAM image using jet colormap
        superimposed_image: Tensor. Superimposed GradGAN image

    """
    # make sure the input image is a tensor (img_height, img_width, 3)
    image = tf.cast(image, tf.float32)

    # default: last conv layer
    if layer_name is None:
        layer_name = find_conv_layer(model)

    # rebuild the pretrained model, add one more output: output of the last conv layer
    gradcam_model = tf.keras.Model(inputs=[model.inputs],
                                   outputs=[model.get_layer(layer_name).output, model.output])

    # get outputs of the last conv layer/the model
    with tf.GradientTape() as tape:
        # add batch dim for 3D image tensor before be fed into the model
        # conv_layer_output: tensor (1, i, j, num_feature_maps k); predictions: tensor (1, n_classes)
        conv_layer_output, predictions = gradcam_model(tf.expand_dims(image, axis=0), training=False)
        if index is None:
            index = tf.math.argmax(predictions[0])  # tensor, e.g. 0, no shape
        # output of the c-th neuron y_c (before softmax): tensor (1,)
        output = predictions[:, index]

    # the gradient of the logit y_c (output before the softmax) for class c, is computed
    # with respect to feature maps A_k of the last conv layer: tensor (1, i, j, k)
    grads = tape.gradient(output, conv_layer_output)

    # remove the dimension of size 1
    conv_layer_output = conv_layer_output[0]  # tensor (i,j,k)
    grads = grads[0]  # tensor (i,j,k)

    # global-average-pooled gradient: tensor (k,)
    pooled_grads = tf.math.reduce_mean(grads, axis=(0, 1))

    # Per broadcasting, pooled_grads tensor (k,) -> (i,j,k)
    # sum along the k axis, headmap tensor (i,j)
    heatmap = tf.math.reduce_sum(pooled_grads * conv_layer_output, axis=-1)
    # apply relu and normalize to [0,1], tensor (i,j)
    heatmap = tf.math.maximum(heatmap, 0)/tf.math.reduce_max(heatmap)

    # from 'jet' RGBA colormap get RGB values in [0,1], ndarray (i,j,3)
    jet_heatmap = plt.get_cmap('jet')(heatmap)[:, :, :3]
    # resize
    image_height, image_width, _ = image.shape
    jet_heatmap = tf.image.resize(jet_heatmap, [image_height, image_width])  # tensor (img_height,img_width,3)
    # superimpose
    superimposed_image = 0.3*jet_heatmap+0.5*image  # tensor (img_height,img_width,3)
    superimposed_image = superimposed_image/tf.math.reduce_max(superimposed_image)  # tensor (img_height,img_width,3)

    return jet_heatmap, superimposed_image
# ------------- GradGAM ------------ #


# ------------- Guided Backpropagation ---------- #
@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, 'float32') * tf.cast(x > 0, 'float32') * dy

    # Forward pass, same value as relu;
    # guided backpropagation: dy = dx_n/dx_l = dx_n/dx_(n-1) * ... * dx_(l+1)/dx_l;
    # dx_n/dx_(l-1) = [dx_n/dx_l > 0] * [x_(l-1) > 0] * dx_(l+1)/dx_l
    return tf.nn.relu(x), grad


def guided_backpropagation(image, model, layer_name=None):

    image = tf.cast(image, tf.float32)

    if layer_name is None:
        layer_name = find_conv_layer(model)

    guided_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output])

    # replace relu with guided_relu
    for layer in guided_model.layers:
        if hasattr(layer, 'activation') and layer.activation == tf.keras.activations.relu:
            layer.activation = guided_relu
    # # Be careful. All 'relu' will be replaced with 'guided relu'.
    # If activation=tf.nn.relu in custom model, this condition layer.activation == tf.keras.activations.relu
    # will not hold. There may be difference between tf.nn.relu and tf.keras.activations.relu. So please use
    # activation='relu' or activation=tf.keras.activations.relu in custom model.
    # One can use guided_model.get_config() to check if activations 'relu' have be replaced with 'guided_relu'

    with tf.GradientTape() as tape:
        image = tf.expand_dims(image, axis=0)  # tensor (1, img_height, img_width, 3)
        tape.watch(image)  # image is a tf.Tensor, which is not watched by default
        outputs = guided_model(image, training=False)  # tensor (1,i,j,num_feature_maps)

    guided_grads = tape.gradient(outputs, image)[0]  # how to know the shape of this gradient?

    return guided_grads
# ------------- Guided Backpropagation ---------- #
