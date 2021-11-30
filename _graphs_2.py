import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def open_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


dataset = 'ucm'  # 'ucm', 'rsicd'

clean = 0.2 if dataset == 'rsicd' else 0.3

probs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

bit = 64
markers = ['s', 'd', 'v', 'o', 'H', 'p', '^']
colors = ['red', 'purple', 'blue', 'green', 'orange', 'brown', 'palevioletred']
models = ['duch', 'duch', 'duch', 'duch', 'duch', 'jdsh', 'djsrh']
tags = ['default', 'default', 'default', 'default', 'default_baseline', 'noisy', 'noisy']
weights = ['normal', 'exp', 'dis', 'ones', 'ones', 'normal', 'normal']
weight_names = ['DUCH-NR-NW', 'DUCH-NR-EW', 'DUCH-NR-DW', 'DUCH-PTC', 'DUCH', 'JDSH', 'DJSRH']

map_names = ['I \u2192 T', 'T \u2192 I']  # ['i2t', 't2i', 'i2i', 't2t']  # ['i2t', 't2i', 'i2i', 't2t', 'avg']
experimnet_names = ['k', 'k', 'k', 'hr', 'hr']
experiment_val = [5, 10, 20, 0, 5]

paths = {'duch': r'/home/george/Code/noisy_captions/checkpoints',
         'jdsh': r'/home/george/Code/jdsh_noisy/checkpoints',
         'djsrh': r'/home/george/Code/jdsh_noisy/checkpoints'}

data = {}

# read data
for weight, tag, model in zip(weights, tags, models):
    for prob in probs:
        token = model + weight + str(prob) + tag
        if model == 'duch':
            folder = '_'.join([dataset, str(bit), tag, str(prob), str(clean), weight])
        if model in ['jdsh', 'djsrh']:
            folder = '_'.join([model, str(bit), dataset, tag, weight, str(prob)]).upper()
        data[token] = open_pkl(os.path.join(paths[model], folder, 'maps_eval.pkl'))

final_table = [['probs'] + probs]


# plotting
fig = plt.figure(figsize=(20, 10))


plot_num = ['(a)', '(b)']
ylims = {'ucm': [0.3, 1.0], 'rsicd': [0.3, 0.9]}

for i, map in enumerate(map_names):
    ax = plt.subplot(1, 2, i+1)

    final_table.append([map])

    for weight, weight_name, tag, model, color, marker in zip(weights, weight_names, tags, models, colors, markers):
        y = []
        for prob in probs:
            token = model + weight + str(prob) + tag
            y.append(data[token][2][i])

        ax.plot(probs, y, label=weight_name, color=color, marker=marker, markersize=12, mew=1, mec='black', linewidth=3)
        final_table.append([weight_name] + y)

    ax.legend(fontsize=24, framealpha=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    #ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(axis='both', which='major', alpha=0.8, linestyle='-')
    ax.grid(axis='both', which='minor', alpha=0.4, linestyle=':')
    ax.set_ylim(ylims[dataset])
    #ax.set_xlim([0.05, 0.5])
    ax.tick_params(axis='both', labelsize=20)
    plt.xlabel('Noise probability\n{}'.format(plot_num[i]), size=24)
    plt.ylabel('mAP@20', size=24)
    plt.title(map.upper(), fontsize=30, fontweight='bold')

#plt.suptitle(dataset.upper(), size=20, weight='medium')
plt.tight_layout()

print(os.path.join('plots', '{}_{}_{}.png'.format('noise', dataset, clean)))
plt.savefig(os.path.join('plots', '{}_{}_{}.png'.format('noise', dataset, clean)))
plt.savefig(os.path.join('plots', '{}_{}_{}.pdf'.format('noise', dataset, clean)))

df = pd.DataFrame(final_table)
print('plots/plots_table_{}.csv'.format(dataset))
df.to_csv('plots/plots_table_{}.csv'.format(dataset), index=False, header=False)