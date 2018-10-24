import os
from shutil import copyfile

print('Started')

i = 0
for path, subdirs, files in os.walk('/'):
    for name in files:
        i += 1
        if 'hostfile' in name:
            src = os.path.join(path, name)
            dst = os.path.join('/logs/', name)
            print('{} >> {}'.format(src, dst))
            copyfile(src, dst)
        if i%5000 == 0:
            print('{} files'.format(i))
print('Finished. {} files total'.format(i))