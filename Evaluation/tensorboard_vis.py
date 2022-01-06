from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


model1 = pd.read_csv('model1.csv', index_col=None, header=None)
model2 = pd.read_csv('model2.csv', index_col=None, header=None)


model1 = model1.to_numpy()
model2 = model2.to_numpy()


plt.rcParams.update({'font.size': 20})
# plt.title(title)
# plt.subplot(3, 1, 1)
# plt.rcParams.update({'font.size': 20})

ax = plt.gca()

ax.set_ylim([0, 0.5])


plt.plot(model1[:, 0], color='crimson')
plt.plot(model1[:, 3], ':', color='crimson')

plt.plot(model2[:, 0], color='deepskyblue')
plt.plot(model2[:, 3], ':', color='deepskyblue')


plt.legend(('1-layer train', '1-layer val',
            '2-layer train (aug)', '2-layer val (aug)'))
plt.xlabel('Epochs')
plt.ylabel('loss')
# plt.title('DFNet')
plt.grid(True)

plt.show()