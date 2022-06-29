import tensorflow as tf
import os
import pandas as pd
import logging
import matplotlib.pyplot as plt

from input_pipeline.preprocessing import preprocess
from visualization.methods import gradcam, guided_backpropagation


def visualize(model, data_dir, run_paths, checkpoint_dir):
    logging.info('------------ Starting visualization for images in test set -------------')

    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    while True:
        try:
            # choose a test image
            image_id = int(input(f"Please enter an image_id, i.e. an integer in [1,103]. Or 'exit' to exit: "))
            image_name = 'IDRiD_' + f'{image_id:03}'

            # use image name to get the corresponding label and prediction
            df = pd.read_csv(os.path.join(run_paths['path_model_id'], 'image_label_prediction_mapping_test.csv'))
            image_label_prediction = df.loc[df['Image name'] == image_name]
            label = image_label_prediction['Retinopathy grade'].values[0]
            prediction = image_label_prediction['Prediction'].values[0]
            logging.info(f'Choose {image_name} with label {label}, prediction {prediction}')

            # load and preprocess test image, same operations in input_pipeline
            image_path = data_dir + '/IDRID_dataset/images/test/' + image_name + '.jpg'
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image, label = preprocess(image, label)  # shape (img_height, img_width,3), RGB [0,1]

            # Image
            logging.info(f'Image')
            plt.figure(dpi=300)
            plt.imshow(image)
            plt.title(f'{image_name} with label {label}, prediction {prediction} ')
            plt.show()

            # GradCAM
            logging.info(f'GradGAM')
            gradcam_image, superimposed_gradcam_image = gradcam(image, model)
            plt.figure(dpi=300)
            plt.imshow(superimposed_gradcam_image)
            plt.title(f'GradGAM for {image_name} with label {label}, prediction {prediction} ')
            plt.show()

            # Guided Backpropagation
            logging.info(f'Guided GradGAM')
            guided_grads = guided_backpropagation(image, model)
            guided_gradcam = guided_grads * gradcam_image
            guided_gradcam = tf.clip_by_value(
                0.25 * ((guided_gradcam - tf.math.reduce_mean(guided_gradcam)) / (tf.math.reduce_std(guided_gradcam) + 1e-7)) + 0.5, 0, 1)
            plt.figure(dpi=300)
            plt.imshow(guided_gradcam)
            plt.title(f'Guided GradGAM for {image_name} with label {label}, prediction {prediction} ')
            plt.show()
        except ValueError:
            logging.info('Exit the visualization')
            break
