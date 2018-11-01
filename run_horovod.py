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

print('='*40, 'Environment Variables', '='*40)
for k, v in os.environ.items():
    print('{}: {}'.format(k, v))
print('='*100)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--module', type=str, default='run_horovod')
    parser.add_argument('--gpus_per_device', type=int, default=0,
                        choices=[i for i in range(101)])
    opts, opts2 = parser.parse_known_args()
    return opts, opts2


def main(opts, opts2):
    proc_per_device = max(opts.gpus_per_device, 1)
    hosts = [host.split(':')[0] for host in worker_hosts+ps_hosts]
    port = worker_hosts[0].split(':')[1]
    n_hosts = len(hosts)
    n_procs = proc_per_device * n_hosts
    hosts = ','.join([host+':{}'.format(proc_per_device) for host in hosts])

    if job_name == 'worker' and task_index == 0:
        # From https://github.com/uber/horovod/blob/master/docs/docker.md
        # mpirun -np 16 -H host1:4,host2:4,host3:4,host4:4 -mca plm_rsh_args "-p 12345" python keras_mnist_advanced.py
        cmd = 'mpirun --verbose -np {np} -H {hosts} -mca plm_rsh_args "-p {port}" python -m {module}'\
              .format(np=n_procs, hosts=hosts, port=port, module=opts.module)
        cmd = ' '.join([cmd]+opts2)
    else:
        cmd = 'bash -c "/usr/sbin/sshd -p {port}; sleep infinity'.format(port=port)

    print('-'*50)
    print('Running command: {}'.format(cmd))
    print('-'*50)
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main(*parse_args())
