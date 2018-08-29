import json
import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from clusterone import get_data_path, get_logs_path

from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

tf.logging.set_verbosity(tf.logging.INFO)
# try:
#     config = os.environ['TF_CONFIG']
#     config = json.loads(config)
#     task = config['task']['type']
#     task_index = config['task']['index']

#     local_ip = 'localhost:' + config['cluster'][task][task_index].split(':')[1]
#     config['cluster'][task][task_index] = local_ip
#     if task == 'chief' or task == 'master':
#         config['cluster']['worker'][task_index] = local_ip
#     os.environ['TF_CONFIG'] = json.dumps(config)
# except:
#     pass

try:
    job_name = os.environ['JOB_NAME']
    print('job_name', job_name)
    task_index = os.environ['TASK_INDEX']
    print('task_index', task_index)
    ps_hosts = os.environ['PS_HOSTS'].strip('[]').split(',')
    print('ps_hosts', ps_hosts)
    worker_hosts = os.environ['WORKER_HOSTS'].strip('[]').split(',')
    print('worker_hosts', worker_hosts)
    TF_CONFIG = {'task': {'type': job_name, 'index': task_index},
                 'cluster': {'chief': [worker_hosts[0]],
                             'worker': worker_hosts,
                             'ps': ps_hosts},
                 'environment': 'cloud'}
    local_ip =  'localhost' + TF_CONFIG['cluster'][job_name][task_index].split(':')[1]
    TF_CONFIG['cluster'][job_name][task_index] = local_ip
    if job_name == 'chief' or job_name == 'master':
        TF_CONFIG['cluster']['worker'][task_index] = local_ip
    os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)
    print(TF_CONFIG)
    print(os.environ['TF_CONFIG'])
except:
    print('*** FAILED ***')

for varname in ['JOB_NAME', 'TASK_INDEX', 'PS_HOSTS', 'WORKER_HOSTS', 'TF_CONFIG']:
    try:
        print(varname, '=', os.environ[varname])
    except:
        print('***CANNOT FIND', varname)

import sys
import time
time.sleep(60)
sys.exit()

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
    parser.add_argument('--cnn', type=str2bool, default=False,
                        help='If true, use CNN. Otherwise, use MLP. Default: False')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Ignored if cnn is False')
    parser.add_argument('--hidden_units', type=int, nargs='*', default=[256, 256])
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

def mlp_model(opts):
    '''Return a MLP Keras model'''
    input_tensor = tf.keras.layers.Input(shape=(784,), name='input')

    temp = input_tensor
    for i, n_units in enumerate(opts.hidden_units):
        temp = tf.keras.layers.Dense(n_units, activation='relu', name='fc'+str(i))(temp)
        temp = tf.keras.layers.Dropout(opts.dropout, name='dropout'+str(i))(temp)
    output = tf.keras.layers.Dense(10, activation='softmax', name='output')(temp)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
    optimizer = tf.keras.optimizers.Adam(lr=opts.learning_rate, decay=opts.learning_decay)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def cnn_model(opts):
    '''Return a CNN Keras model'''
    input_tensor = tf.keras.layers.Input(shape=(784,), name='input')

    temp = tf.keras.layers.Reshape([28, 28, 1], name='input_image')(input_tensor)
    for i, n_units in enumerate(opts.hidden_units):
        temp = tf.keras.layers.Conv2D(n_units, kernel_size=opts.kernel_size, strides=(2, 2),
                                      activation='relu', name='cnn'+str(i))(temp)
        temp = tf.keras.layers.Dropout(opts.dropout, name='dropout'+str(i))(temp)
    temp = tf.keras.layers.GlobalAvgPool2D(name='average')(temp)
    output = tf.keras.layers.Dense(10, activation='softmax', name='output')(temp)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
    optimizer = tf.keras.optimizers.Adam(lr=opts.learning_rate, decay=opts.learning_decay)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def main(opts):
    if opts.fashion:
        data = read_data_sets(opts.data_dir,
                              source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    else:
        data = read_data_sets(opts.data_dir)

    if opts.cnn:
        model = cnn_model(opts)
    else:
        model = mlp_model(opts)
    config = tf.estimator.RunConfig(
                model_dir=opts.log_dir,
                save_summary_steps=1,
                save_checkpoints_steps=1000,
                keep_checkpoint_max=5,
                log_step_count_steps=100)
    classifier = tf.keras.estimator.model_to_estimator(model, model_dir=opts.log_dir, config=config)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                         x={'input': data.train.images},
                         y=tf.keras.utils.to_categorical(data.train.labels.astype(np.int32), 10).astype(np.float32),
                         num_epochs=None,
                         batch_size=opts.batch_size,
                         shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                         x={'input': data.test.images},
                         y=tf.keras.utils.to_categorical(data.test.labels.astype(np.int32), 10).astype(np.float32),
                         num_epochs=1,
                         shuffle=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=1e6)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=None,
                                      start_delay_secs=0,
                                      throttle_secs=30)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

if __name__ == '__main__':
    opts = get_args()
    main(opts)
