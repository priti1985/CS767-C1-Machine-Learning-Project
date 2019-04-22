import tensorflow as tf

from cnn_model import network


def model_fn(features, labels, mode, params):
    """
    Creates model_fn for Tensorflow estimator. This function takes features and input, and
    is responsible for the creation and processing of the Tensorflow graph for training, prediction and evaluation.

    Expected feature: {'image': image tensor }

    :param features: dictionary of input features
    :param labels: dictionary of ground truth labels
    :param mode: graph mode
    :param params: params to configure model
    :return: Estimator spec dependent on mode
    """
    learning_rate = params['learning_rate']
    image_input = features['image']

    age_logits, logits = network(image_input, labels, mode)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return get_prediction_spec(age_logits, logits)

    joint_loss = get_loss(age_logits, logits, labels)

    with tf.name_scope('gender_accuracy'):
        gender_accuracy = tf.metrics.accuracy(
            labels=labels['gender'], predictions=tf.argmax(logits, axis=1),name='gender_acc_op')
        tf.summary.scalar('gender_accuracy', gender_accuracy[1])

    with tf.name_scope('age_accuracy'):
        age_accuracy = tf.metrics.accuracy(labels=labels['age'], predictions=tf.argmax(age_logits, axis=1),name='age_acc_op')
        tf.summary.scalar('age_accuracy', age_accuracy[1])

    train_hook_list = []
    train_tensors_log = {'gender_accuracy': gender_accuracy[1],
                         'age_accuracy': age_accuracy[1],
                         'loss': joint_loss,
                         'global_step': tf.train.get_global_step()}
    train_hook_list.append(tf.train.LoggingTensorHook(
        tensors=train_tensors_log, every_n_iter=200))


    if mode == tf.estimator.ModeKeys.TRAIN:
        return get_training_spec(learning_rate, joint_loss,train_hook_list)

    else:
        return get_eval_spec(logits, age_logits, labels, joint_loss,gender_accuracy,age_accuracy)


def get_prediction_spec(age_logits, logits):
    """
    Creates estimator spec for prediction

    :param age_logits: logits of age task
    :param logits: logits of gender task
    :return: Estimator spec
    """
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "age_class": tf.argmax(input=age_logits, name='age_class', axis=1),
        "age_prob": tf.nn.softmax(age_logits, name='age_prob'),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    export_outputs = {
        'prediction': tf.estimator.export.PredictOutput(predictions)
    }
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions,export_outputs=export_outputs)


def get_loss(age_logits, gender_logits, labels):
    """
    Creates joint loss function

    :param age_logits: logits of age
    :param gender_logits: logits of gender task
    :param labels: ground-truth labels of age and gender
    :return: joint loss of age and gender
    """
    gender_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels['gender'], logits=gender_logits)
    age_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels['age'], logits=age_logits)
    joint_loss = gender_loss + age_loss
    return joint_loss


def get_eval_spec(gender_logits, age_logits, labels, loss,gender_accuracy,age_accuracy):
    """
    Creates eval spec for tensorflow estimator
    :param gender_logits: logits of gender task
    :param age_logits: logits of age task
    :param labels: ground truth labels for age and gender
    :param loss: loss op
    :return: Eval estimator spec
    """
    eval_metric_ops = {
        "gender_accuracy":gender_accuracy,
        "age_accuracy":age_accuracy
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_ops,evaluation_hooks=None)


def get_training_spec(learning_rate, joint_loss,train_hook_list):
    """
    Creates training estimator spec

    :param learning rate for optimizer
    :param joint_loss: loss op
    :return: Training estimator spec
    """

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gender_train_op = optimizer.minimize(
        loss=joint_loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=joint_loss, train_op=gender_train_op,training_hooks=train_hook_list)


def serving_fn():
    receiver_tensor = {
        'image': tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    }

    features = {
        'image': tf.image.resize_images(receiver_tensor['image'], [224, 224])
    }

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)