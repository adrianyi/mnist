import os
from argparse import ArgumentParser
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
from clusterone import get_data_path, get_logs_path

tf.logging.set_verbosity(tf.logging.INFO)
try:
    job_name = os.environ['JOB_NAME']
    task_index = int(os.environ['TASK_INDEX'])
    ps_hosts = os.environ['PS_HOSTS'].split(',')
    worker_hosts = os.environ['WORKER_HOSTS'].split(',')
except KeyError as ex:
    job_name = None
    task_index = 0
    ps_hosts = []
    worker_hosts = []
print('job_name:', job_name)
print('task_index:', task_index)
print('ps_hosts:', ps_hosts)
print('worker_hosts:', worker_hosts)

n_gpus = len([x for x in list_local_devices() if x.device_type == 'GPU'])
distribution = tf.contrib.distribute.MirroredStrategy(num_gpus_per_worker=n_gpus)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise IOError('Boolean value expected (i.e. yes/no, true/false, y/n, t/f, 1/0).')


def get_args():
    """Return parsed args"""
    parser = ArgumentParser()
    parser.add_argument('--local_data_dir', type=str, default='data/',
                        help='Path to local data directory')
    parser.add_argument('--local_log_dir', type=str, default='logs/',
                        help='Path to local log directory')
    parser.add_argument('--fashion', type=str2bool, default=False,
                        help='Use Fashion MNIST data')

    # Model params
    parser.add_argument('--hidden_units', type=int, nargs='*', default=[32, 64])
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=512)

    # Training params
    parser.add_argument('--eval_secs', type=int, default=120,
                        help='throttle_secs for EvalSpec')

    args = parser.parse_args()

    args.data_dir = get_data_path(dataset_name='*/*',
                                  local_root=args.local_data_dir,
                                  local_repo='',
                                  path='')
    args.log_dir = get_logs_path(root=args.local_log_dir)

    return args


def device_and_target():
    # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
    # Don't set a device.
    if job_name is None:
        print("Running single-machine training")
        return None, ""

    # Otherwise we're running distributed TensorFlow.
    print("%s.%d  -- Running distributed training" % (job_name, task_index))
    if task_index is None or task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    if ps_hosts is None or ps_hosts == "":
        raise ValueError("Must specify an explicit `ps_hosts`")
    if worker_hosts is None or worker_hosts == "":
        raise ValueError("Must specify an explicit `worker_hosts`")

    cluster_spec = tf.train.ClusterSpec({
        "ps": ps_hosts,
        "worker": worker_hosts,
    })
    server = tf.train.Server(
        cluster_spec, job_name=job_name, task_index=task_index)
    if job_name == "ps":
        server.join()

    worker_device = "/job:worker/task:{}".format(task_index)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    return (
        tf.train.replica_device_setter(
              worker_device=worker_device,
              cluster=cluster_spec),
        server.target,
    )


# def parser(example):
#     features = {'image': tf.FixedLenFeature([784], dtype=tf.float32),
#                 'label': tf.FixedLenFeature([], dtype=tf.int64)}
#     data = tf.parse_single_example(
#         serialized=example,
#         features=features)
#     return tf.cast(data['image'], tf.float32), tf.cast(data['label'], tf.int32)
#
#
# def dataset_fn(filenames, batch_size=512, train=False):
#     """Input function to be used for Estimator class"""
#     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=2)
#     dataset = dataset.map(parser)
#     if train:
#         dataset = tf.data.Dataset.apply(dataset, tf.contrib.data.shuffle_and_repeat(5 * opts.batch_size, count=None))
#     dataset = dataset.batch(batch_size=batch_size)
#     return dataset


class IteratorInitializerHook(tf.train.SessionRunHook):
    """From https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0"""
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        self.iterator_initializer_func(session)


def input_fn(x, y, batch_size=512, train=True):
    """Input function to be used for Estimator class"""
    data_x = tf.data.Dataset.from_tensor_slices(x)
    data_y = tf.data.Dataset.from_tensor_slices(y)
    dataset = tf.data.Dataset.zip((data_x, data_y))

    if train:
        dataset = tf.data.Dataset.apply(dataset,
                                        tf.contrib.data.shuffle_and_repeat(5 * batch_size, count=None))
    dataset = dataset.batch(batch_size=batch_size)

    return dataset


def cnn_net(input_tensor, opts):
    """Return logits output from CNN net"""
    temp = tf.reshape(input_tensor, shape=(-1, 28, 28, 1), name='input_image')
    for i, n_units in enumerate(opts.hidden_units):
        temp = tf.layers.conv2d(temp, filters=n_units, kernel_size=(3, 3), strides=(2, 2),
                                activation=tf.nn.relu, name='cnn'+str(i))
        temp = tf.layers.dropout(temp, rate=opts.dropout)
    temp = tf.reduce_mean(temp, axis=(2, 3), keepdims=False, name='average')
    return tf.layers.dense(temp, 10)


def tower_fn(data, opts):
    features, labels = data[0], data[1]
    # logits = distribution.call_for_each_tower(cnn_net, features)
    # logits = tf.group(distribution.unwrap(logits))
    logits = cnn_net(features, opts)
    pred = tf.cast(tf.argmax(logits, axis=1), tf.int64)
    acc, acc_op = tf.metrics.accuracy(labels, pred)

    cent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                          logits=logits,
                                                          name='cross_entropy')
    loss = tf.reduce_mean(cent, name='loss')

    optimizer = tf.train.AdamOptimizer(learning_rate=opts.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

    # return pred, loss, train_op
    return train_op


def main(opts):
    device, target = device_and_target()
    print('Device:', device)
    print('Target:', target)

    # filenames = glob.glob(os.path.join(opts.data_dir, 'train*.tfrecords'))
    # train_dataset_fn = partial(dataset_fn, filenames, batch_size=opts.batch_size, train=True)

    data = read_data_sets(opts.data_dir,
                          one_hot=False,
                          fake_data=False)

    train_x_ph = tf.placeholder(tf.float32, [None, 784])
    train_y_ph = tf.placeholder(tf.int32, [None])

    train_dataset_fn = partial(input_fn, train_x_ph, train_y_ph, batch_size=opts.batch_size, train=True)

    # train_input_fn, train_iter_hook = get_inputs(data.train.images,
    #     tf.keras.utils.to_categorical(data.train.labels.astype(np.int32), 10).astype(np.float32),
    #     batch_size=opts.batch_size,
    #     train=True)
    # eval_input_fn, eval_iter_hook = get_inputs(data.test.images,
    #     tf.keras.utils.to_categorical(data.test.labels.astype(np.int32), 10).astype(np.float32),
    #     batch_size=opts.batch_size,
    #     train=False)

    with distribution.scope():
        iterator = distribution.distribute_dataset(train_dataset_fn).make_initializable_iterator()
        tower_train_ops = distribution.call_for_each_tower(tower_fn, iterator.get_next(), opts)
        train_op = tf.group(distribution.unwrap(tower_train_ops))

    with tf.train.MonitoredTrainingSession(
             master=target,
             is_chief=(task_index == 0),
             checkpoint_dir=opts.log_dir,
             log_step_count_steps=50) as sess:
        sess.run(iterator.initializer, feed_dict={train_x_ph: data.train.images,
                                                  train_y_ph: data.train.labels.astype(np.int32)})
        local_step = 0
        while not sess.should_stop():
            local_step += 1
            loss_value, _, global_step = sess.run(['loss:0', train_op, tf.train.get_or_create_global_step()])
            if local_step % 50 == 0:
                tf.logging.info('{} {} {} {}'.format(task_index, local_step, global_step, loss_value))


if __name__ == '__main__':
    main(get_args())
