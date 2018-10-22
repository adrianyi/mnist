import json
import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from clusterone import get_data_path, get_logs_path

from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

tf.logging.set_verbosity(tf.logging.INFO)
try:
    job_name = os.environ['JOB_NAME']
    task_index = int(os.environ['TASK_INDEX'])
    ps_hosts = os.environ['PS_HOSTS'].split(',')
    worker_hosts = os.environ['WORKER_HOSTS'].split(',')

    if job_name == 'ps':
        ps_hosts[task_index] = 'localhost:' + ps_hosts[task_index].split(':')[1]
    else:
        worker_hosts[task_index] = 'localhost:' + worker_hosts[task_index].split(':')[1]
except KeyError as ex:
    job_name = None
    task_index = 0
    ps_hosts = []
    worker_hosts = []

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise IOError('Boolean value expected (i.e. yes/no, true/false, y/n, t/f, 1/0).')

def get_args():
    '''Return parsed args'''
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

    opts = parser.parse_args()

    opts.data_dir = get_data_path(dataset_name = 'adrianyi/mnist-data',
                                 local_root = opts.local_data_dir,
                                 local_repo = '',
                                 path = '')
    opts.log_dir = get_logs_path(root = opts.local_log_dir)

    return opts

def device_and_target():
  # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
  # Don't set a device.
  if job_name is None:
    print("Running single-machine training")
    return (None, "")

  # Otherwise we're running distributed TensorFlow.
  print("%s.%d  -- Running distributed training"%(job_name, task_index))
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

def cnn_net(input_tensor):
    '''Return logits output from CNN net'''
    temp = tf.reshape(input_tensor, shape=(-1, 28, 28, 1), name='input_image')
    for i, n_units in enumerate(opts.hidden_units):
        temp = tf.layers.conv2d(temp, filters=n_units, kernel_size=(3, 3), strides=(2, 2),
                                activation=tf.nn.relu, name='cnn'+str(i))
        temp = tf.layers.dropout(temp, rate=opts.dropout)
    temp = tf.reduce_mean(temp, axis=(2,3), keepdims=False, name='average')
    return tf.layers.dense(temp, 10)

def model(features, labels):
    logits = cnn_net(features)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy'), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=opts.learning_rate)
    return logits, loss, optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

def main(opts):
    device, target = device_and_target()
    print('Device:', device)
    print('Target:', target)

    if opts.fashion:
        data = read_data_sets(opts.data_dir,
                              source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    else:
        data = read_data_sets(opts.data_dir)

    ea_custom_getter = tf.contrib.opt.ElasticAverageCustomGetter("/job:worker/task:{}".format(task_index))
    with tf.device(device), tf.variable_scope('',custom_getter=ea_custom_getter):
        features = data.train.images
        labels = data.train.labels.astype(np.int32)

        features_placeholder = tf.placeholder(features.dtype, features.shape)
        labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        dataset = tf.data.Dataset.apply(dataset, tf.contrib.data.shuffle_and_repeat(5*opts.batch_size, count=None))
        dataset = dataset.batch(batch_size=opts.batch_size).repeat()
        iterator = dataset.make_initializable_iterator()

        x, y = iterator.get_next()
        # logits, loss, train_op = model(x, y)
        logits = cnn_net(x)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
        with tf.device("/job:worker/task:{}".format(task_index)):
            optimizer = tf.train.MomentumOptimizer(learning_rate=opts.learning_rate, momentum=0.01)
            optimizer = tf.contrib.opt.ElasticAverageOptimizer(optimizer, num_worker=len(worker_hosts), ea_custom_getter=ea_custom_getter)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
        hooks = [optimizer.make_session_run_hook(is_chief=(task_index == 0), task_index=task_index)]

    with tf.train.MonitoredTrainingSession(
        master=target,
        is_chief=(task_index == 0),
        checkpoint_dir=opts.log_dir,
        log_step_count_steps=50) as sess:
        sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                                  labels_placeholder: labels})
        local_step = 0
        while not sess.should_stop():
            local_step += 1
            loss_value, _, global_step = sess.run([loss, train_op, tf.train.get_or_create_global_step()])
            if local_step%10 == 0:
                tf.logging.info('{} {} {} {}'.format(task_index, local_step, global_step, loss_value))

if __name__ == '__main__':
    opts = get_args()
    main(opts)
