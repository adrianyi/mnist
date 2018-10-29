import glob
import json
import os
from argparse import ArgumentParser

import tensorflow as tf
from clusterone import get_data_path, get_logs_path

from tensorflow.python.client.device_lib import list_local_devices

tf.logging.set_verbosity(tf.logging.INFO)
try:
    print(os.environ)
    job_name = os.environ['JOB_NAME']
    task_index = int(os.environ['TASK_INDEX'])
    ps_hosts = os.environ['PS_HOSTS'].split(',')
    worker_hosts = os.environ['WORKER_HOSTS'].split(',')
    TF_CONFIG = {'task': {'type': job_name, 'index': task_index},
                 'cluster': {'chief': [worker_hosts[0]],
                             'worker': worker_hosts,
                             'ps': ps_hosts},
                 'environment': 'cloud'}
    local_ip = 'localhost:' + TF_CONFIG['cluster'][job_name][task_index].split(':')[1]
    if (job_name == 'chief') or (job_name == 'worker' and task_index == 0):
        job_name = 'chief'
        TF_CONFIG['task']['type'] = 'chief'
        TF_CONFIG['cluster']['worker'][0] = local_ip
    TF_CONFIG['cluster'][job_name][task_index] = local_ip
    os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)
    print(TF_CONFIG)
except KeyError as ex:
    job_name = 'local'
    task_index = 0
    ps_hosts = []
    worker_hosts = []
print('job name:', job_name)
print('task_index:', task_index)
print('ps_hosts:', ps_hosts)
print('worker_hosts:', worker_hosts)


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
    parser.add_argument('--user_name', type=str, default='adrian-stable/')
    parser.add_argument('--dataset_name', type=str, default='mnist-tfrecord-1')

    # Model params
    parser.add_argument('--hidden_units', type=int, nargs='*', default=[32, 64])
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=512)

    # Training params
    parser.add_argument('--eval_secs', type=int, default=120,
                        help='throttle_secs for EvalSpec')

    # Distribution strategy params
    parser.add_argument('--distribution_strategy', type=str, default=None,
                        choices=['mirrored', 'collectiveallreduce'])
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='None means use auto-detect. 0 means use CPU.')

    args = parser.parse_args()

    args.data_dir = get_data_path(dataset_name='/'.join([args.user_name, args.dataset_name]),
                                  local_root=args.local_data_dir,
                                  local_repo='',
                                  path='')
    args.log_dir = get_logs_path(root=args.local_log_dir)
    args.log_dir = os.path.join(args.log_dir, '{}-{}'.format(job_name, task_index))

    return args


def get_device_names(num_gpus=None):
    if num_gpus is None:
        devices = [x.name for x in list_local_devices() if x.device_type == 'GPU']
        if len(devices) < 1:
            return [x.name for x in list_local_devices()]
    elif num_gpus == 0:
        return ['/device:CPU:0']
    return ['/device:GPU:{}'.format(i) for i in range(num_gpus)]


def tfparser(example):
    features = {'image': tf.FixedLenFeature([784], dtype=tf.float32),
                'label': tf.FixedLenFeature([], dtype=tf.int64)}
    data = tf.parse_single_example(
        serialized=example,
        features=features)

    return tf.cast(data['image'], tf.float32), tf.cast(data['label'], tf.int32)


def get_inputs(filenames, batch_size=512, train=True):
    """Returns input function and and iterator initializer hook"""

    def input_fn():
        """Input function to be used for Estimator class"""
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=2)
        dataset = dataset.map(tfparser)
        if train:
            dataset = tf.data.Dataset.apply(dataset, tf.contrib.data.shuffle_and_repeat(5*batch_size, count=None))
        dataset = dataset.batch(batch_size=batch_size)

        dataset = dataset

        return dataset

    return input_fn


def cnn_net(input_tensor, opts):
    """Return logits output from CNN net"""
    temp = tf.reshape(input_tensor, shape=(-1, 28, 28, 1), name='input_image')
    for i, n_units in enumerate(opts.hidden_units):
        temp = tf.layers.conv2d(temp, filters=n_units, kernel_size=(3, 3), strides=(2, 2),
                                activation=tf.nn.relu, name='cnn'+str(i))
        temp = tf.layers.dropout(temp, rate=opts.dropout)
    temp = tf.reduce_mean(temp, axis=(2, 3), keepdims=False, name='average')
    return tf.layers.dense(temp, 10)


def get_model_fn(opts):
    optimizer = tf.train.AdamOptimizer(learning_rate=opts.learning_rate)

    def model_fn(features, labels, mode):
        logits = cnn_net(features, opts)
        pred = tf.cast(tf.argmax(logits, axis=1), tf.int64)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions={'logits': logits, 'pred': pred})

        cent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
        loss = tf.reduce_mean(cent, name='loss')

        metrics = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=pred, name='accuracy')}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return model_fn


def main(opts):
    devices = get_device_names(opts.num_gpus)
    if opts.num_gpus is None:
        opts.num_gpus = len([x for x in devices if 'GPU' in x])

    if opts.distribution_strategy is None:
        print('Not using distribution strategy')
        distribution = None
    elif opts.distribution_strategy == 'mirrored':
        print('Using MirroredStrategy with num_gpus={}'.format(opts.num_gpus))
        distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=opts.num_gpus)
    elif opts.distribution_strategy == 'collectiveallreduce':
        print('Using CollectiveAllReduceStrategy with num_gpus_per_worker={}'.format(opts.num_gpus))
        distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(num_gpus_per_worker=opts.num_gpus)
    else:
        raise NotImplementedError('Distribution strategy not implemented: {}'.format(opts.distribution_strategy))

    train_filenames = glob.glob(os.path.join(opts.data_dir, 'train*.tfrecords'))
    test_filenames = glob.glob(os.path.join(opts.data_dir, 'test*.tfrecords'))

    train_input_fn = get_inputs(train_filenames,
                                batch_size=opts.batch_size,
                                train=True)
    eval_input_fn = get_inputs(test_filenames,
                               batch_size=opts.batch_size,
                               train=False)

    runconfig = tf.estimator.RunConfig(
        model_dir=opts.log_dir,
        save_summary_steps=50,
        save_checkpoints_steps=50,
        keep_checkpoint_max=2,
        log_step_count_steps=10,
        train_distribute=distribution,
        eval_distribute=distribution)
    estimator = tf.estimator.Estimator(
        model_fn=get_model_fn(opts),
        config=runconfig)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=1e6)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=None,
                                      throttle_secs=10)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    try:
        main(get_args())
    except Exception as ex:
        print('='*100)
        print('{}-{}'.format(job_name, task_index))
        print('='*100)
        raise ex
