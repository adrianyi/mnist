from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import time

from tensorflow.python.client.device_lib import list_local_devices


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
num_gpus = len([x for x in list_local_devices() if x.device_type == 'GPU'])
num_procs = max(num_gpus, 1)
print('job name:', job_name)
print('task_index:', task_index)
print('ps_hosts:', ps_hosts)
print('worker_hosts:', worker_hosts)
print('Num GPUs:', num_gpus)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_procs_per_node', type=int, default=num_procs)
    args = parser.parse_args()
    return args


def touch(fname, times=None):
    with open(fname, 'w'):
        os.utime(fname, times)


def main(opts):
    filename = 'hostfile'

    if job_name == 'worker' and task_index == 0:
        touch(filename)
        with open(filename, 'a') as f:
            lines = ['{host} slots={np} max-slots={np}'.format(host=host, np=opts.num_procs_per_node)
                     for host in worker_hosts + ps_hosts]
            f.write('\n'.join(lines))
    else:
        while not os.path.exists(filename):
            time.sleep(1)


if __name__ == '__main__':
    opts = parse_args()
    main(opts)