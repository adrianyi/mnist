from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import shutil
import subprocess
import time

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

is_chief = job_name == 'worker' and task_index == 0

print('='*40, 'Environment Variables', '='*40)
for k, v in os.environ.items():
    print('{}: {}'.format(k, v))
print('='*100)


def str2bool(v):
    """"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise IOError('Boolean value expected (i.e. yes/no, true/false, y/n, t/f, 1/0).')


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--module', type=str, default='run_horovod')
    parser.add_argument('--gpus_per_device', type=int, default=0,
                        choices=[i for i in range(101)])
    parser.add_argument('--sleep_infinity', type=str2bool, default=False)
    opts, opts2 = parser.parse_known_args()
    return opts, opts2


def make_ssh_or_wait():
    if is_chief:
        if os.path.exists('~/.ssh'):
            print('~/.ssh exists, removing directory')
            shutil.rmtree('~/.ssh')
        if os.path.exists('/data/.ssh'):
            print('Copying /data/.ssh to ~/.ssh')
            shutil.copytree('/data/.ssh', '~/.ssh')
        print('Generating ssh-key & config')
        cmd = ' && '.join([
            'mkdir /data/.ssh/',
            'ssh-keygen -f /data/.ssh/id_rsa -t rsa -N ""',
            'mv /data/.ssh/id_rsa.pub /data/.ssh/authorized_keys',
            'chmod 600 /data/.ssh/authorized_keys',
            'echo "Host *" >> /data/.ssh/config',
            'echo " IdentityFile ~/.ssh/id_rsa" >> /data/.ssh/config',
            'cp -r /data/.ssh ~/.ssh'
        ])
        print('Running command: {}'.format(cmd))
        subprocess.call(cmd, shell=True)
        time.sleep(5)
    else:
        print('Waiting for chief node to create ssh keys')
        while not os.path.exists('/data/.ssh/id_rsa'):
            time.sleep(3)
        time.sleep(1)
        cmd = ' && '.join([
            'cp -r /data/.ssh ~/.ssh',
            '/usr/sbin/sshd -p 5000'
        ])
        print('Running command: {}'.format(cmd))
        subprocess.call(cmd, shell=True)


def main(opts, opts2):
    proc_per_device = max(opts.gpus_per_device, 1)
    hosts = [host.split(':')[0] for host in worker_hosts+ps_hosts]
    port = worker_hosts[0].split(':')[1]
    n_hosts = len(hosts)
    n_procs = proc_per_device * n_hosts
    hosts = ','.join([host+':{}'.format(proc_per_device) for host in hosts])

    if is_chief:
        # From https://github.com/uber/horovod/blob/master/docs/docker.md
        # mpirun -np 16 -H host1:4,host2:4,host3:4,host4:4 -mca plm_rsh_args "-p 12345" python keras_mnist_advanced.py
        cmd = 'mpirun --verbose -np {np} -H {hosts} -mca plm_rsh_args "-p {port}" python -m {module}'\
              .format(np=n_procs, hosts=hosts, port=port, module=opts.module)
        cmd = ' '.join([cmd]+opts2)
    else:
        cmd = 'while true; do sleep 60; echo $(date); done'

    print('-'*50)
    print('Running command: {}'.format(cmd))
    print('-'*50)
    subprocess.call(cmd, shell=True)
    if opts.sleep_infinity:
        i = 0
        while True:
            print('Slept for {}min, sleeping for 1 more minute'.format(i))
            time.sleep(60)
            i += 1


if __name__ == '__main__':
    make_ssh_or_wait()
    main(*parse_args())
