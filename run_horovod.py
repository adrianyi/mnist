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

n_gpus = 1
n_hosts = len(worker_hosts)
n_procs = n_gpus * n_hosts
hosts = ','.join([host+':{}'.format(n_gpus) for host in worker_hosts])

cmd = 'mpirun --verbose -np {np} -H {hosts} python mnist_horovod.py'.format(np=n_procs, hosts=hosts)
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