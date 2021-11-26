import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def open_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


dataset = 'rsicd'  # 'ucm', 'rsicd'

clean = 0.2 if dataset == 'rsicd' else 0.3

probs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

bit = 64
tags = ['default', 'default', 'default', 'default', 'default_baseline']
weights = ['normal', 'exp', 'dis', 'ones', 'ones']
weight_names = ['normal_weights', 'exponential_weights', 'discrete_weights', 'no_noise_reduction', 'baseline']

map_names = ['i2t', 't2i']  # ['i2t', 't2i', 'i2i', 't2t']  # ['i2t', 't2i', 'i2i', 't2t', 'avg']
experimnet_names = ['k', 'k', 'k', 'hr', 'hr']
experiment_val = [5, 10, 20, 0, 5]

paths = {'unhd': r'/home/george/Code/noisy_captions/checkpoints'}

data = {}

# read data
for weight, tag in zip(weights, tags):
    for prob in probs:
        token = weight + str(prob) + tag
        folder = '_'.join([dataset, str(bit), tag, str(prob), str(clean), weight])
        data[token] = open_pkl(os.path.join(paths['unhd'], folder, 'maps_eval.pkl'))


final_table = [['probs'] + probs]


# plotting
fig = plt.figure(figsize=(12, 6))


for i, map in enumerate(map_names):
    ax = plt.subplot(1, 2, i+1)

    final_table.append([map])

    for weight, weight_name, tag in zip(weights, weight_names, tags):
        y = []
        for prob in probs:
            token = weight + str(prob) + tag
            y.append(data[token][2][i])

        ax.plot(probs, y, label=weight_name)
        final_table.append([weight_name] + y)

    ax.legend(fontsize=16)
    ax.grid(axis='both', which='major', alpha=0.8, linestyle='-')
    ax.grid(axis='both', which='minor', alpha=0.4, linestyle=':')
    plt.xlabel('noise probability', fontsize=16)
    plt.ylabel('mAP@20', fontsize=16)
    plt.title(map.upper())

plt.suptitle(dataset.upper(), size=20, weight='medium')
plt.tight_layout()

print(os.path.join('plots', '{}_{}_{}.png'.format('noise', dataset, clean)))
plt.savefig(os.path.join('plots', '{}_{}_{}.png'.format('noise', dataset, clean)))
plt.savefig(os.path.join('plots', '{}_{}_{}.pdf'.format('noise', dataset, clean)))

df = pd.DataFrame(final_table)
print('plots/plots_table_{}.csv'.format(dataset))
df.to_csv('plots/plots_table_{}.csv'.format(dataset), index=False, header=False)