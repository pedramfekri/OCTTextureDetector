from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


model1 = pd.read_csv('model1.csv', index_col=None, header=None)
model2 = pd.read_csv('model2.csv', index_col=None, header=None)
model3 = pd.read_csv('model3.csv', index_col=None, header=None)
model4 = pd.read_csv('model4.csv', index_col=None, header=None)
model5 = pd.read_csv('model5.csv', index_col=None, header=None)

model1 = model1.to_numpy()
model2 = model2.to_numpy()
model3 = model3.to_numpy()
model4 = model4.to_numpy()
model5 = model5.to_numpy()

plt.rcParams.update({'font.size': 20})
# plt.title(title)
# plt.subplot(3, 1, 1)
# plt.rcParams.update({'font.size': 20})

ax = plt.gca()

ax.set_ylim([0, 0.0015])


plt.plot(model1[:, 0], color='crimson')
plt.plot(model1[:, 3], ':', color='crimson')

plt.plot(model3[:, 0], color='black')
plt.plot(model3[:, 3], ':', color='black')

plt.plot(model2[:, 0], color='deepskyblue')
plt.plot(model2[:, 3], ':', color='deepskyblue')


plt.plot(model4[:, 0], color='green')
plt.plot(model4[:, 3], ':', color='green')

plt.plot(model5[:, 0], color='peru')
plt.plot(model5[:, 3], ':', color='peru')

plt.legend(('1-layer train', '1-layer val',
            '2-layer train (aug)', '2-layer val (aug)',
            '2-layer train', '2layer val',
            '3-layer train (aug)', '3-layer val (aug)',
            '3-layer train', '3-layer val'))
plt.xlabel('Epochs')
plt.ylabel('Mean Square Error')
# plt.title('DFNet')
plt.grid(True)

plt.show()