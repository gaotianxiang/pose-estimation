import tensorflow as tf
from data import create_dataset
import os
import time

path = '/home/tianxiang/dataset/mpii/tfr_processed'
tfrecords = [os.path.join(path, fn) for fn in os.listdir(path) if fn.startswith('train')]

dtst = create_dataset(tfrecords, 16, is_train=True)

print('start')
start = time.time()
for i, x in enumerate(dtst):
    pass
end = time.time()

print('end')
print('time used {}'.format(end - start))
