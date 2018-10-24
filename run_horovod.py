from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import subprocess

try:
    job_name = os.environ['JOB_NAME']
    task_index = int(os.environ['TASK_INDEX'])
    ps_hosts = os.environ['PS_HOSTS'].split(',')
    worker_hosts = os.environ['WORKER_HOSTS'].split(',')
except KeyError as ex:
    job_name = None
    task_index = 0
    ps_hosts = []
    worker_hosts = ['localhost']
print('job name:', job_name)
print('task_index:', task_index)
print('ps_hosts:', ps_hosts)
print('worker_hosts:', worker_hosts)

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpus_per_device', type=int, default=0,
                        choices=[i for i in range(101)])
    opts, opts2 = parser.parse_known_args()
    return opts, opts2

def main(opts, opts2):
    proc_per_device = max(opts.gpus_per_device, 1)
    hosts = [host.split(':')[0] for host in worker_hosts+ps_hosts]
    n_hosts = len(hosts)
    n_procs = proc_per_device * n_hosts
    hosts = ','.join([host+':{}'.format(proc_per_device) for host in hosts])

    cmd = 'mpirun --verbose -np {np} -H {hosts} python mnist_horovod.py'.format(np=n_procs, hosts=hosts)
    cmd = ' '.join([cmd]+opts2)
    # Command
    # mpirun -np 4 \
    #     -H localhost:4 \
    #     -bind-to none -map-by slot \
    #     -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    #     -mca pml ob1 -mca btl ^openib \
    #     python train.py
    print('-'*50)
    print('Running command: {}'.format(cmd))
    print('-'*50)
    subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    opts, opts2 = parse_args()
    main(opts, opts2)