import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def open_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


dataset = 'ucm'

probs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

bit = 64
tag = 'test'
preset = 'clean'

map_names = ['i2t', 't2i']  # ['i2t', 't2i', 'i2i', 't2t']  # ['i2t', 't2i', 'i2i', 't2t', 'avg']
experimnet_names = ['k', 'k', 'k', 'hr', 'hr']
experiment_val = [5, 10, 20, 0, 5]

paths = {'unhd': r'/home/george/Code/noisy_captions/checkpoints'}

data = {}

# read data
for prob in probs:
    folder = '_'.join([dataset, str(bit), tag, str(prob), preset])
    data[prob] = open_pkl(os.path.join(paths['unhd'], folder, 'maps_eval.pkl'))

# plotting
fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(1, 1, 1)

for i, map in enumerate(map_names):

    y = []
    for prob in probs:
        y.append(data[prob][2][i])

    ax.plot(probs, y, label=map)

ax.legend(fontsize=16)
ax.grid()
plt.title(dataset.upper(), size=20, weight='medium')
plt.xlabel('noise probability', fontsize=16)
plt.ylabel('mAP@20', fontsize=16)
plt.tight_layout()

print(os.path.join('plots', '{}_{}.png'.format('noise', dataset)))
plt.savefig(os.path.join('plots', '{}_{}.png'.format('noise', dataset)))
