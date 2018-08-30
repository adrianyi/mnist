import json
import os

def get_tf_config():
    try:
        try:
            TF_CONFIG = json.loads(os.environ['TF_CONFIG'])
            job_name = TF_CONFIG['task']['type']
            task_index = TF_CONFIG['task']['index']
            print('Using TF_CONFIG variable')
        except KeyError as ex:
            print(str(ex))
            job_name = os.environ['JOB_NAME']
            print('job_name', job_name)
            task_index = int(os.environ['TASK_INDEX'])
            print('task_index', task_index)
            ps_hosts = os.environ['PS_HOSTS'].split(',')
            print('ps_hosts', ps_hosts)
            worker_hosts = os.environ['WORKER_HOSTS'].split(',')
            print('worker_hosts', worker_hosts)
            print('Building TF_CONFIG variable')
            TF_CONFIG = {'task': {'type': job_name, 'index': task_index},
                        'cluster': {'chief': [worker_hosts[0]],
                                    'worker': worker_hosts,
                                    'ps': ps_hosts},
                        'environment': 'cloud'}
            print('TF_CONFIG', TF_CONFIG)
            if job_name == 'worker' and task_index == 0:
                TF_CONFIG['task']['type'] = 'chief'
                print('TF_CONFIG chief', TF_CONFIG)
        TF_CONFIG['cluster'][job_name][task_index] = 'localhost:5000'
        print('TF_CONFIG', TF_CONFIG)
        if job_name == 'chief':
            TF_CONFIG['cluster']['worker'][task_index] = 'localhost:5000'
            print('TF_CONFIG', TF_CONFIG)
        elif job_name == 'worker' and task_index == 0:
            TF_CONFIG['cluster']['chief'] = ['localhost:5000']
            print('TF_CONFIG', TF_CONFIG)
        return TF_CONFIG
    except KeyError as ex:
        print(str(ex))
        print('No TF_CONFIG, local mode')
        return {}
