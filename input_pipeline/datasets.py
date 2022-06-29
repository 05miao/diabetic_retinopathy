import gin
import logging
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from sklearn.model_selection import train_test_split

from input_pipeline.preprocessing import preprocess, augment


@gin.configurable
def load(name, data_dir, n_classes, run_paths):
    load.data_dir = data_dir
    if name == "idrid":
        logging.info(f"------------ Preparing dataset {name}... -----------")

        # ---------- Choose problem type: binary or 5-class classification ----------
        if n_classes == 2:
            logging.info('Binary classification. Class 1 : labels 0,1; class 2: labels 2 and up')
        elif n_classes == 5:
            logging.info('5-class classification. Class 0 ~ 4.')
        else:
            raise ValueError('Number of classes should be 2 or 5')

        # ---------- Load image from path ----------
        @ tf.function
        def load_image(image_path, label):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            return image, label

        # ---------- Test set ----------
        path_test = data_dir + '/IDRID_dataset/images/test/'
        # get the image_label_mapping
        image_label_mapping_test = pd.read_csv(data_dir + '/IDRID_dataset/labels/test.csv')[
            ['Image name', 'Retinopathy grade']]
        # when binary classification, labels should be modified. Class 1 : labels 0,1; class 2: labels 2 and up
        if n_classes == 2:
            image_label_mapping_test["Retinopathy grade"] = \
                (image_label_mapping_test["Retinopathy grade"] > 1).astype(int)
        logging.info(f"Test set -\n"
                     f"{image_label_mapping_test['Retinopathy grade'].value_counts().sort_index().to_string()}")
        # save to csv. When evaluating, class predictions of test images  will also be appended to this csv file
        image_label_mapping_test.to_csv(os.path.join(run_paths['path_model_id'], 'image_label_prediction_mapping_test.csv'), index=False)
        labels_test = image_label_mapping_test['Retinopathy grade']
        # image paths of test set
        image_paths_test = [path_test + image_name + '.jpg' for image_name in image_label_mapping_test['Image name']]
        # test set
        ds_test = tf.data.Dataset.from_tensor_slices((image_paths_test, labels_test)).map(load_image)

        # ---------- The full training set, which will be split into training/validation set ----------
        path_train_full = data_dir + '/IDRID_dataset/images/train/'
        # get the image_label_mapping
        image_label_mapping_train_full = pd.read_csv(data_dir + '/IDRID_dataset/labels/train.csv')[
            ['Image name', 'Retinopathy grade']]
        # when binary classification, labels should be modified. Class 1 : labels 0,1; class 2: labels 2 and up
        if n_classes == 2:
            image_label_mapping_train_full["Retinopathy grade"] = \
                (image_label_mapping_train_full["Retinopathy grade"] > 1).astype(int)
        # shuffle and split the full training set into training/validation set
        logging.info('Splitting the original training set into training/validation set (8:2)')
        image_label_mapping_train, image_label_mapping_val = train_test_split(
            image_label_mapping_train_full, test_size=0.2, random_state=1)

        # ---------- Training set ----------
        sample_amounts = image_label_mapping_train['Retinopathy grade'].value_counts().sort_index()
        logging.info(f'imbalanced training set before oversampling -\n{sample_amounts.to_string()}')
        # oversampling for the imbalanced training set
        resample_size = sample_amounts.max()
        image_label_mapping_train = image_label_mapping_train.groupby('Retinopathy grade').apply(
            lambda g: g.sample(n=resample_size, replace=len(g) < resample_size, random_state=1))
        # shuffle the training set
        image_label_mapping_train = image_label_mapping_train.sample(
            frac=1, random_state=1).reset_index(drop=True)
        logging.info(f"Balanced training set after oversampling -\n"
                     f"{image_label_mapping_train['Retinopathy grade'].value_counts().sort_index().to_string()}")
        # labels of training set
        labels_train = image_label_mapping_train['Retinopathy grade']
        # image paths of training set
        image_paths_train = [path_train_full + image_name + '.jpg' for image_name in
                             image_label_mapping_train['Image name']]
        # training set
        ds_train = tf.data.Dataset.from_tensor_slices((image_paths_train, labels_train)).map(load_image)

        # ---------- Validation set ----------
        logging.info(f"Validation set -\n"
                     f"{image_label_mapping_val['Retinopathy grade'].value_counts().sort_index().to_string()}")
        # labels of validation set
        labels_val = image_label_mapping_val['Retinopathy grade']
        # image paths of validation set
        image_paths_val = [path_train_full + image_name + '.jpg' for image_name in
                           image_label_mapping_val['Image name']]
        # validation set
        ds_val = tf.data.Dataset.from_tensor_slices((image_paths_val, labels_val)).map(load_image)

        # ---------- ds_info ----------
        ds_info = {'n_train': len(image_label_mapping_train.index),
                   'input_shape': (2848, 4288, 3),
                   'n_classes': n_classes}

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...\n    ")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        ds_info = {'n_train': ds_info.splits['train'].num_examples,
                   'input_shape': ds_info.features["image"].shape,
                   'n_classes': ds_info.features["label"].num_classes}

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )

        ds_info = {'n_train': ds_info.splits['train'].num_examples,
                   'input_shape': ds_info.features["image"].shape,
                   'n_classes': ds_info.features["label"].num_classes}

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info['n_train'])
    ds_train = ds_train.repeat()
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test
