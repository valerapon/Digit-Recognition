import numpy as np
import pandas as pd
from PIL import Image



data_csv = pd.read_csv('Digit-recognition/tmp/train.csv')

target = data_csv['label'].to_numpy(dtype=int)
data = data_csv.iloc[:,1:].to_numpy(dtype=int)


f = open('Digit-recognition/tmp/data_train.txt', 'w')
for i in range(0, len(target), 128):
        f.write(' '.join([str(j) for j in target[i:i + 128]]) + ' ')
        f.write(' '.join([str(j) for j in np.reshape(data[i:i + 128, :], -1)]) + '\n')
f.close()

