import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
from clusterone import get_data_path, get_logs_path

from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

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
    # if job_name == 'ps':
    #     ps_hosts[task_index] = 'localhost:' + ps_hosts[task_index].split(':')[1]
    # else:
    #     worker_hosts[task_index] = 'localhost:' + worker_hosts[task_index].split(':')[1]
except KeyError as ex:
    job_name = None
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
    '''Return parsed args'''
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
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
    parser.add_argument('--eval_steps', type=int, default=1000)

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

class IteratorInitializerHook(tf.train.SessionRunHook):
    '''From https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0'''
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        self.iterator_initializer_func(session)

def cnn_net(input_tensor):
    '''Return logits output from CNN net'''
    temp = tf.reshape(input_tensor, shape=(-1, 28, 28, 1), name='input_image')
    for i, n_units in enumerate(opts.hidden_units):
        temp = tf.layers.conv2d(temp, filters=n_units, kernel_size=(3, 3), strides=(2, 2),
                                activation=tf.nn.relu, name='cnn'+str(i))
        temp = tf.layers.dropout(temp, rate=opts.dropout)
    temp = tf.reduce_mean(temp, axis=(2,3), keepdims=False, name='average')
    return tf.layers.dense(temp, 10)

def main(opts):
    # Initiate Horovod
    hvd.init()

    if opts.fashion:
        data = read_data_sets(opts.data_dir,
                              source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    else:
        data = read_data_sets(opts.data_dir)

    def inputfn_and_initializer(x, y, train=False):
        iterator_initializer_hook = IteratorInitializerHook()

        def input_fn():
            x_placeholder = tf.placeholder(x.dtype, x.shape)
            y_placeholder = tf.placeholder(y.dtype, y.shape)

            dataset = tf.data.Dataset.from_tensor_slices((x_placeholder, y_placeholder))
            if train:
                dataset = tf.data.Dataset.apply(dataset, tf.contrib.data.shuffle_and_repeat(5*opts.batch_size, count=None))
            dataset = dataset.batch(batch_size=opts.batch_size)
            iterator = dataset.make_initializable_iterator()
            iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer,
                                                                                        feed_dict={x_placeholder: x,
                                                                                                   y_placeholder: y})
            return iterator.get_next()
        return input_fn, iterator_initializer_hook

    def model_fn(features, labels, mode):
        logits = cnn_net(features)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy'), name='loss')
        optimizer = tf.train.MomentumOptimizer(learning_rate=opts.learning_rate * hvd.size(), momentum=0.01)
        # Horovod distributed optimizer
        optimizer = hvd.DistributedOptimizer(optimizer)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

        pred = tf.cast(tf.argmax(logits, axis=1), tf.int64)

        metrics = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=pred, name='accuracy')}

        return tf.estimator.EstimatorSpec(mode=mode,
                                            predictions=pred,
                                            train_op=train_op,
                                            loss=loss,
                                            eval_metric_ops=metrics)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())

    runconfig = tf.estimator.RunConfig(
        model_dir=(opts.log_dir if hvd.rank() == 0 else None),
        save_summary_steps=50,
        save_checkpoints_steps=50,
        keep_checkpoint_max=2,
        log_step_count_steps=10,
        session_config=session_config)
    estimator = tf.estimator.Estimator(
        model_dir=(opts.log_dir if hvd.rank() == 0 else None),
        model_fn=model_fn,
        config=runconfig)
    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

    train_input_fn, train_iter_hook = inputfn_and_initializer(data.train.images, data.train.labels.astype(np.int32), train=True)
    eval_input_fn, eval_iter_hook = inputfn_and_initializer(data.test.images, data.test.labels.astype(np.int32), train=False)

    while True:
        estimator.train(
            input_fn=train_input_fn,
            steps=opts.eval_steps // hvd.size(),
            hooks=[train_iter_hook, bcast_hook])
        
        eval_results = estimator.evaluate(input_fn=eval_input_fn)
        print(eval_results)

if __name__ == '__main__':
    opts = get_args()
    main(opts)
