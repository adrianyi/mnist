import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from clusterone import get_data_path, get_logs_path

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.logging.set_verbosity(tf.logging.INFO)
try:
    config = os.environ['TF_CONFIG']
    config = json.loads(config)
    task = config['task']['type']
    task_index = config['task']['index']

    local_ip = 'localhost:' + config['cluster'][task][task_index].split(':')[1]
    config['cluster'][task][task_index] = local_ip
    if task == 'chief' or task == 'master':
        config['cluster']['worker'][task_index] = local_ip
    os.environ['TF_CONFIG'] = json.dumps(config)
except:
    job_name = None
    task_index = 0
    ps_hosts = None
    worker_hosts = None

def get_args():
    '''Return parsed args'''
    parser = ArgumentParser()
    parser.add_argument('--local_data_dir', type=str, default='data/',
                        help='Path to local data directory')
    parser.add_argument('--local_log_dir', type=str, default='logs/',
                        help='Path to local log directory')
    # Model params
    parser.add_argument('--hidden_units', type=int, nargs='*', default=[256, 32])
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    opts = parser.parse_args()

    opts.data_dir = get_data_path(dataset_name = '*/*',
                                 local_root = opts.local_data_dir,
                                 local_repo = '',
                                 path = '')
    opts.log_dir = get_logs_path(root = opts.local_log_dir)

    return opts

def print_data(data_dir, n=100):
    '''Print first n files in data_dir'''
    print('Printing {n} files in {d}'.format(n=n, d=data_dir))
    i = 0
    for path, subdirs, files in os.walk(PATH):
        for name in files:
            print(os.path.join(path, name))
            if i > n:
                return

def main(opts):
    data = read_data_sets(opts.data_dir,
                          one_hot=False,
                          fake_data=False)
    feature_columns = [tf.feature_column.numeric_column('x', shape=[28, 28])]

    config = tf.estimator.RunConfig(
                model_dir=opts.log_dir,
                save_summary_steps=500,
                save_checkpoints_steps=500,
                keep_checkpoint_max=5,
                log_step_count_steps=10)
    classifier = tf.estimator.DNNClassifier(
                     feature_columns=feature_columns,
                     hidden_units=opts.hidden_units,
                     optimizer=tf.train.AdamOptimizer(opts.learning_rate),
                     n_classes=10,
                     dropout=opts.dropout,
                     model_dir=opts.log_dir)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                         x={'x': data.train.images},
                         y=data.train.labels.astype(np.int32),
                         num_epochs=None,
                         batch_size=opts.batch_size,
                         shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                         x={'x': data.test.images},
                         y=data.test.labels.astype(np.int32),
                         num_epochs=1,
                         shuffle=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=1e6)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      start_delay_secs=30,
                                      throttle_secs=300)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

if __name__ == '__main__':
    opts = get_args()
    print_data(opts.data_dir, 100)
    main(opts)
