import tensorflow as tf


class ConfusionMatrix(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """

    def __init__(self, n_classes, name='confusion_matrix', **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.n_classes = n_classes
        # The constructor uses the add_weight() method to create the variables
        # needed to keep track of the metric’s state over multiple batches
        self.total_confusion_matrix = self.add_weight("total_confusion_matrix",
                                                      shape=(n_classes, n_classes), initializer="zeros")

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred):
        # y_true: integer labels e.g. [0, 1];
        # y_pred: already be transformed into integer labels in eval.py
        # dtype=tf.float32 seems to be needed. Otherwise, there will be a Value Error:
        # Tensor conversion requested dtype float32 for Tensor with dtype int32
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32,
                                                    num_classes=self.n_classes)
        self.total_confusion_matrix.assign_add(confusion_matrix)

    def result(self):
        return self.total_confusion_matrix

    def metrics_from_confusion_matrix(self):
        """returns precision, sensitivity, f1"""
        # confusion matrix C
        confusion_matrix = self.total_confusion_matrix
        # diagonal part of confusion matrix = [C_ii]
        diag_part = tf.linalg.diag_part(confusion_matrix)
        # label i, precision[i] = C_ii/(sum of C_ki from k=0 to k=num_classes-1)
        precision = diag_part / (tf.reduce_sum(confusion_matrix, 0) + tf.constant(1e-15))
        # label i, recall[i] = C_ii/(sum of C_ij from j=0 to k=num_classes-1)
        sensitivity = diag_part / (tf.reduce_sum(confusion_matrix, 1) + tf.constant(1e-15))
        f1 = 2 * precision * sensitivity / (precision + sensitivity + tf.constant(1e-15))
        return precision, sensitivity, f1
