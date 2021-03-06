import logging
import os
import tensorflow as tf
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from metrics import ConfusionMatrix
from visualization.visualize import visualize


def evaluate(model, ds_test, data_dir, n_classes, run_paths, checkpoint_dir):

    """

    Args:
        model (keras.Model): model
        ds_test (tf.data.Dataset): test set
        data_dir (str): parent directory of e.g. IDRID_dataset
        n_classes (int): can be 2 or 5
        run_paths (dict): contains the info about the directory structure of 'experiments'. This directory
          structure is automatically generated by utils_params.gen_run_folder().
        checkpoint_dir (str): e.g. 'xxx/ckpts/train'. used to restore a training checkpoint,.
          When starting evaluation soon after training, this dir can be run_paths['path_ckpt_train'].
          When just starting evaluation without training, this dir must be manually specified.

    """

    logging.info("\n------------ Starting evaluation ------------")

    confusion_matrix = ConfusionMatrix(n_classes)
    ckpt = tf.train.Checkpoint(model=model)
    logging.info(f"restore checkpoint: \n{tf.train.latest_checkpoint(checkpoint_dir)}")
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    # read the csv file with image-label mapping of test set, predictions will be added into it as a new column
    df = pd.read_csv(os.path.join(run_paths['path_model_id'], 'image_label_prediction_mapping_test.csv'))
    # create an empty list, which will be extended with predictions of each batch
    list_predictions = []
    for test_images, test_labels in ds_test:  # batched test set, tensor (batch_size, img_height, img_width, 3)
        test_predictions = model(test_images, training=False)
        # test_predictions: tensor (batch_size, n_classes). Not normalized because there is no softmax layer
        # in the model. Needed to be transformed into integer labels e.g. [[2.5, 1.3], [1.1, 2.7]] to [0,1]
        test_predictions = tf.math.argmax(test_predictions, axis=1)  # tensor (batch_size,)
        # confusion matrix
        confusion_matrix.update_state(test_labels, test_predictions)
        list_predictions += test_predictions.numpy().tolist()
    # get all predictions after for-loop
    df['Prediction'] = list_predictions
    # sorted by labels and predictions
    df = df.sort_values(by=['Retinopathy grade', 'Prediction'])
    # save csv file
    df.to_csv(os.path.join(run_paths['path_model_id'], 'image_label_prediction_mapping_test.csv'), index=False)
    logging.info('\n------ Evaluation statistics')
    logging.info(f'{df.to_string()}')

    # confusion matrix
    logging.info(f"Confusion matrix: \n{confusion_matrix.result().numpy()}")

    # precision, sensitivity, f1
    precision, sensitivity, f1 = confusion_matrix.metrics_from_confusion_matrix()
    logging.info(f"Precision: {precision}, Sensitivity: {sensitivity}, F1: {f1}")

    # show the confusion matrix
    plt.figure(dpi=300)
    sn.heatmap(confusion_matrix.result().numpy(), annot=True)
    plt.title('Evaluation - Confusion matrix')
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # Deep visualization
    visualize(model, data_dir, run_paths, checkpoint_dir)
