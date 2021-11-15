import json
import os
import h5py
import pickle

import numpy as np
import torch
import logging
import random
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt


def read_json(file_name, suppress_console_info=False):
    """
    Read JSON

    :param file_name: input JSON path
    :param suppress_console_info: toggle console printing
    :return: dictionary from JSON
    """
    with open(file_name, 'r') as f:
        data = json.load(f)
        if not suppress_console_info:
            print("Read from:", file_name)
    return data


def get_image_file_names(data, suppress_console_info=False):
    """
    Get list of image file names

    :param data: original data from JSON
    :param suppress_console_info: toggle console printing
    :return: list of strings (file names)
    """

    file_names = []
    for img in data['images']:
        file_names.append(img["filename"])
    if not suppress_console_info:
        print("Total number of files:", len(file_names))
    return file_names


def get_labels(data, suppress_console_info=False):
    """
    Get list of labels

    :param data: original data from JSON
    :param suppress_console_info: toggle console printing
    :return: list ints (labels)
    """

    labels = []
    for img in data['images']:
        labels.append(img["classcode"])
    if not suppress_console_info:
        print("Total number of labels:", len(labels))
    return labels


def get_captions(data, suppress_console_info=False):
    """
    Get list of formatted captions

    :param data: original data from JSON
    :return: list of strings (captions)
    """

    def format_caption(string):
        return string.replace('.', '').replace(',', '').replace('!', '').replace('?', '').lower()

    captions = []
    augmented_captions_rb = []
    augmented_captions_bt_prob = []
    augmented_captions_bt_chain = []
    for img in data['images']:
        for sent in img['sentences']:
            captions.append(format_caption(sent['raw']))
            try:
                augmented_captions_rb.append(format_caption(sent['aug_rb']))
            except:
                pass
            try:
                augmented_captions_bt_prob.append(format_caption(sent['aug_bt_prob']))
            except:
                pass
            try:
                augmented_captions_bt_chain.append(format_caption(sent['aug_bt_chain']))
            except:
                pass
    if not suppress_console_info:
        print("Total number of captions:", len(captions))
        print("Total number of augmented captions RB:", len(augmented_captions_rb))
        print("Total number of augmented captions BT (prob):", len(augmented_captions_bt_prob))
        print("Total number of augmented captions BT (chain):", len(augmented_captions_bt_chain))
    return captions, augmented_captions_rb, augmented_captions_bt_prob, augmented_captions_bt_chain


def add_prefix_to_filename(file_name, prefix):
    """
    Adds prefix to the file name

    :param file_name: file name string
    :param prefix: prefix
    :return: file name string with prefix
    """
    bn = os.path.basename(file_name)
    dn = os.path.dirname(file_name)
    return os.path.join(dn, prefix + bn)


def write_json(file_name, data):
    """
    Write dictionary to JSON file

    :param file_name: output path
    :param data: dictionary
    :return: None
    """
    bn = os.path.basename(file_name)
    dn = os.path.dirname(file_name)
    name, ext = os.path.splitext(bn)
    file_name = os.path.join(dn, name + '.json')
    with open(file_name, 'w') as f:
        f.write(json.dumps(data, indent='\t'))
    print("Written to:", file_name)


def print_parsed_args(parsed):
    """
    Print parsed arguments

    :param parsed: Namespace of parsed arguments
    :return: None
    """
    vs = vars(parsed)
    print("Parsed arguments: ")
    for k, v in vs.items():
        print("\t" + str(k) + ": " + str(v))


def write_hdf5(out_file, data, dataset_name):
    """
    Write to h5 file

    :param out_file: file name
    :param data: data to write
    :return:
    """
    bn = os.path.basename(out_file)
    dn = os.path.dirname(out_file)
    name, ext = os.path.splitext(bn)
    out_file = os.path.join(dn, name + '.h5')
    with h5py.File(out_file, 'w') as hf:
        print("Saved as '.h5' file to", out_file)
        hf.create_dataset(dataset_name, data=data)


def read_hdf5(file_name, dataset_name, normalize=False):
    """
    Read from h5 file

    :param file_name: file name
    :param dataset_name: dataset name
    :param normalize: normalize loaded values
    :return:
    """
    with h5py.File(file_name, 'r') as hf:
        print("Read from:", file_name)
        data = hf[dataset_name][:]
        if normalize:
            data = (data - data.mean()) / data.std()
        return data


def write_pickle(path, data):
    """
    Write pickle

    :param path: path
    :param data: data to write
    :return:
    """
    dn = os.path.dirname(path)
    if not os.path.exists(dn):
        os.makedirs(dn)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def select_idxs(seq_length, n_to_select, n_from_select, seed=42):
    """
    Select n_to_select indexes from each consequent n_from_select indexes from range with length seq_length, split
    selected indexes to separate arrays

    Example:

    seq_length = 20
    n_from_select = 5
    n_to_select = 2

    input, range of length seq_length:
    range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    sequences of length n_from_select:
    sequences = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]

    selected n_to_select elements from each sequence
    selected = [[0, 4], [7, 9], [13, 14], [16, 18]]

    output, n_to_select lists of length seq_length / n_from_select:
    output = [[0, 7, 13, 16], [4, 9, 14, 18]]

    :param seq_length: length of sequence, say 10
    :param n_to_select: number of elements to select
    :param n_from_select: number of consequent elements
    :return:
    """
    random.seed(seed)
    idxs = [[] for _ in range(n_to_select)]
    for i in range(seq_length // n_from_select):
        ints = random.sample(range(n_from_select), n_to_select)
        for j in range(n_to_select):
            idxs[j].append(i * n_from_select + ints[j])
    return idxs


def select_k_idxs_from_pop(k, population, seed=42):
    random.seed(seed)
    return sorted(random.sample(range(len(population)), k))


def shuffle_file_names_list(file_names):
    """
    We need to shuffle the data because resnet has batch normalization

    :param file_names: img file names list
    :return: shuffled list and permutation
    """
    fn_np = np.array(file_names)
    perm = np.random.permutation(len(fn_np))
    fn_np_perm = fn_np[perm]
    return fn_np_perm, perm


def unshuffle_embeddings(embeddings, permutation):
    """
    Reorder embeddings array back to original

    :param embeddings: embeddings
    :param permutation: permutation of original shuffle
    :return: embeddings with original data order
    """

    def get_idx(i):
        idx = np.where(permutation == i)[0][0]
        return idx

    result = np.zeros_like(embeddings)

    for i in range(len(result)):
        idx = get_idx(i)
        result[i] = embeddings[idx]

    return result


def calc_hamming_dist(B1, B2):
    """
    Hamming distance

    :param B1: binary codes
    :param B2: binary codes
    :return:
    """
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    """
    calculate MAPs

    :param qB: query binary codes
    :param rB: response binary codes
    :param query_label: labels of query
    :param retrieval_label: labels of response
    :param k: k
    :return:
    """
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_map_rad(qB, rB, query_label, retrieval_label):
    """
    calculate MAPs, in regard to hamming radius

    :param qB: query binary codes
    :param rB: response binary codes
    :param query_label: labels of query
    :param retrieval_label: labels of response
    :return:
    """

    num_query = qB.shape[0]  # length of query (each sample from query compared to retrieval samples)
    num_bit = qB.shape[1]  # length of hash code
    P = torch.zeros(num_query, num_bit + 1)  # precisions (for each sample)

    # for each sample from query calculate precision and recall
    for i in range(num_query):
        # gnd[j] == 1 if same class, otherwise 0, ground truth
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # tsum (TP + FN): total number of samples of the same class
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)  # hamming distances from qB[i, :] (current sample) to retrieval samples
        # tmp[k,j] == 1 if hamming distance to retrieval sample j is less or equal to k (distance), 0 otherwise
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        # total (TP + FP): total[k] is count of distances less or equal to k (from query sample to retrieval samples)
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.0001  # replace zeros with 0.1 to avoid division by zero
        # select only same class samples from tmp (ground truth masking, only rows where gnd == 1 proceed further)
        t = gnd * tmp
        # count (TP): number of true (correctly selected) samples of the same class for any given distance k
        count = t.sum(dim=-1)
        p = count / total  # TP / (TP + FP)
        P[i] = p
    P = P.mean(dim=0)
    return P


def pr_curve(qB, rB, query_label, retrieval_label, tqdm_label=''):
    """
    Calculate PR curve, each point - hamming radius

    :param qB: query hash code
    :param rB: retrieval hash codes
    :param query_label: query label
    :param retrieval_label: retrieval label
    :param tqdm_label: label for tqdm's output
    :return:
    """
    if tqdm_label != '':
        tqdm_label = 'PR-curve ' + tqdm_label

    num_query = qB.shape[0]  # length of query (each sample from query compared to retrieval samples)
    num_bit = qB.shape[1]  # length of hash code
    P = torch.zeros(num_query, num_bit + 1)  # precisions (for each sample)
    R = torch.zeros(num_query, num_bit + 1)  # recalls (for each sample)

    # for each sample from query calculate precision and recall
    for i in tqdm(range(num_query), desc=tqdm_label):
        # gnd[j] == 1 if same class, otherwise 0, ground truth
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # tsum (TP + FN): total number of samples of the same class
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)  # hamming distances from qB[i, :] (current sample) to retrieval samples
        # tmp[k,j] == 1 if hamming distance to retrieval sample j is less or equal to k (distance), 0 otherwise
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        # total (TP + FP): total[k] is count of distances less or equal to k (from query sample to retrieval samples)
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.0001  # replace zeros with 0.1 to avoid division by zero
        # select only same class samples from tmp (ground truth masking, only rows where gnd == 1 proceed further)
        t = gnd * tmp
        # count (TP): number of true (correctly selected) samples of the same class for any given distance k
        count = t.sum(dim=-1)
        p = count / total  # TP / (TP + FP)
        r = count / tsum  # TP / (TP + FN)
        P[i] = p
        R[i] = r
    # mask to calculate P mean value (among all query samples)
    # mask = (P > 0).float().sum(dim=0)
    # mask = mask + (mask == 0).float() * 0.001
    # P = P.sum(dim=0) / mask
    # mask to calculate R mean value (among all query samples)
    # mask = (R > 0).float().sum(dim=0)
    # mask = mask + (mask == 0).float() * 0.001
    # R = R.sum(dim=0) / mask
    P = P.mean(dim=0)
    R = R.mean(dim=0)
    return P.cpu().numpy().tolist(), R.cpu().numpy().tolist()


def p_top_k(qB, rB, query_label, retrieval_label, K, tqdm_label=''):
    """
    P@K curve

    :param qB: query hash code
    :param rB: retrieval hash codes
    :param query_label: query label
    :param retrieval_label: retrieval label
    :param K: K's for curve
    :param tqdm_label: label for tqdm's output
    :return:
    """
    if tqdm_label != '':
        tqdm_label = 'AP@K ' + tqdm_label

    num_query = qB.shape[0]
    PK = torch.zeros(len(K)).to(qB.device)

    for i in tqdm(range(num_query), desc=tqdm_label):
        # ground_truth[j] == 1 if same class (if at least 1 same label), otherwise 0, ground truth
        ground_truth = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # count of samples, that shall be retrieved
        tp_fn = ground_truth.sum()
        if tp_fn == 0:
            continue

        hamm_dist = calc_hamming_dist(qB[i, :], rB).squeeze()

        # for each k in K
        for j, k in enumerate(K):
            k = min(k, retrieval_label.shape[0])
            _, sorted_indexes = torch.sort(hamm_dist)
            retrieved_indexes = sorted_indexes[:k]
            retrieved_samples = ground_truth[retrieved_indexes]
            PK[j] += retrieved_samples.sum() / k

    PK = PK / num_query

    return PK.cpu().numpy().tolist()


def logger():
    """
    Instantiate logger

    :return:
    """
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    log_name = 'log.txt'
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    txt_log = logging.FileHandler(os.path.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger


def retrieval2png(cfg_params, qB, rB, qL, rL, qI, rI, k=20, tag='', file_tag='UNHD'):
    print('Visualizing retrieval for:', tag)

    def get_retrieved_info(qB, rB, qL, rL, qI, rI, k):

        i = 50

        ham_dist = calc_hamming_dist(qB[i, :], rB).squeeze().detach().cpu()
        ham_dist_sorted, idxs = torch.sort(ham_dist)

        ham_dist_sorted_k = ham_dist_sorted[:k].cpu().numpy()
        idxs_k = idxs[:k].cpu().numpy()

        q_idx = qI[i].cpu().numpy()
        r_idxs = rI[idxs_k].cpu().numpy()
        q_lab = np.argmax(qL[i].cpu().numpy())
        r_labs = np.argmax(rL[idxs_k].cpu().numpy(), axis=1)

        return ham_dist_sorted_k, q_idx, r_idxs, q_lab, r_labs

    def load_img_txt():
        data = read_json(cfg_params[1], True)
        file_names = get_image_file_names(data, True)
        img_paths = [os.path.join(cfg_params[2], i) for i in file_names]
        captions, _, _, _ = get_captions(data, True)
        return img_paths, captions

    def get_retrieval_dict(img_paths, captions, q_idx, r_idxs, tag):
        d = {'tag': tag}
        if tag.startswith('I'):
            d['q'] = img_paths[q_idx]
            d['o'] = captions[q_idx]  # captions[q_idx*5]
        else:
            d['q'] = captions[q_idx]  # captions[q_idx*5]
            d['o'] = img_paths[q_idx]
        if tag.endswith('I'):
            d['r'] = [img_paths[r_idx] for r_idx in r_idxs]
        else:
            d['r'] = [captions[r_idx] for r_idx in r_idxs]  # [captions[r_idx*5] for r_idx in r_idxs]

        return d

    def plot_retrieval(d, tag, file_tag, q_lab, r_labs, qI, rI):

        def set_spines_color_width(ax, color, width):
            plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.spines['bottom'].set_color(color)
            ax.spines['top'].set_color(color)
            ax.spines['right'].set_color(color)
            ax.spines['left'].set_color(color)
            ax.spines['bottom'].set_linewidth(width)
            ax.spines['top'].set_linewidth(width)
            ax.spines['right'].set_linewidth(width)
            ax.spines['left'].set_linewidth(width)

        def get_q_rs():
            if tag == 'I2T':
                return Image.open(d['q']), d['r'], d['o']
            elif tag == 'T2I':
                return d['q'], [Image.open(i) for i in d['r']], Image.open(d['o'])
            elif tag == 'I2I':
                return Image.open(d['q']), [Image.open(i) for i in d['r']], d['o']
            elif tag == 'T2T':
                return d['q'], d['r'], Image.open(d['o'])

        colors = ['green' if q_lab == r_lab else 'red' for r_lab in r_labs]

        def print_results():
            print()
            print('Query:')
            print(q_idx, d['q'], d['o'])
            print()
            print('Retrieval:')
            for i, r in zip(r_idxs, d['r']):
                print(i, r)

        q, rs, o = get_q_rs()

        print_results()

        # figure size
        if tag.endswith('I'):
            fig = plt.figure(figsize=((len(rs) + 1) * 2.5, 4))
            subplots = len(rs) + 1
        else:
            fig = plt.figure(figsize=(12, 4))
            subplots = 2

        # plot query
        ax = fig.add_subplot(1, subplots, 1)
        ax.set_title('Query (idx:' + str(qI) + ')')
        if tag.startswith('I'):
            set_spines_color_width(ax, 'black', 3)
            plt.imshow(q)
            ax.text(0, 250, o)
        else:
            #ax.axis([0, len(rs), 0, len(rs)])
            plt.axis('off')
            plt.imshow(o)
            ax.text(0, 250, '(idx:' + str(qI) + ') ' + q)

        # plot responses
        if tag.endswith('I'):
            for i, r in enumerate(rs):
                ax = fig.add_subplot(1, subplots, 2 + i)
                ax.set_title('Response (idx:' + str(rI[i]) + ') ' + str(i + 1))
                set_spines_color_width(ax, colors[i], 3)
                plt.imshow(r)
        else:
            ax = fig.add_subplot(1, subplots, 2)
            for i, r in enumerate(rs):
                ax.axis([0, len(rs), 0, len(rs)])
                plt.axis('off')
                ax.text(0, i, '(idx:' + str(rI[i]) + ') ' + r, color=colors[i])

        # plt.tight_layout()
        plt.savefig(os.path.join('plots', ''.join([tag, file_tag, '.png'])))

    ham_dist_sorted_k, q_idx, r_idxs, q_lab, r_labs = get_retrieved_info(qB, rB, qL, rL, qI, rI, k)
    img_paths, captions = load_img_txt()

    d = get_retrieval_dict(img_paths, captions, q_idx, r_idxs, tag)

    plot_retrieval(d, tag, file_tag, q_lab, r_labs, q_idx, r_idxs)

    return d


def top_k_hists(qBX, qBY, rBX, rBY, k=10, model=''):
    print('Building top@k histograms')

    def top_k_hist_data(qB, rB, k):
        n = len(qB)
        d = {}
        for i in range(n):
            ham_dist = calc_hamming_dist(qB[i, :], rB).squeeze().detach().cpu()
            ham_dist_sorted, idxs = torch.sort(ham_dist)

            ham_dist_sorted_k = ham_dist_sorted[:k].cpu().numpy()
            values, counts = np.unique(ham_dist_sorted_k, return_counts=True)
            for v, c in zip(values.astype(int), counts):
                if v in d:
                    d[v] += c
                else:
                    d[v] = c

        x = list(range(max(d.keys())))
        y = [d[j] if j in d else 0 for j in range(max(d.keys()))]

        return x, y, n

    def plot_top_k_hist(x, y, n, tag, ax):
        rects = ax.bar(x, y, width=1)
        scale = max([rect.get_height() for rect in rects])
        for rect in rects:
            h = rect.get_height()
            if h < scale * 0.1:
                txt_offset = int(scale * 0.05)
            else:
                txt_offset = - int(scale * 0.05)
            ax.annotate('Mean: {:.1f}'.format(h / n), xy=(rect.get_x() + rect.get_width() / 2 - 0.25, h + txt_offset),
                        weight='bold', color='red')
        plt.title(tag.upper(), size=20, weight='medium')
        plt.grid(axis='y')
        ax.set_xticks(x)

    i2t = top_k_hist_data(qBX, rBY, k)
    t2i = top_k_hist_data(qBY, rBX, k)
    i2i = top_k_hist_data(qBX, rBX, k)
    t2t = top_k_hist_data(qBY, rBY, k)

    data = [i2t, t2i, i2i, t2t]
    tags = ['i2t', 't2i', 'i2i', 't2t']

    fig = plt.figure(figsize=(16, 8))
    for i, (tag, d) in enumerate(zip(tags, data)):
        ax = fig.add_subplot(2, 2, i + 1)
        plot_top_k_hist(*d, ', '.join([tag, model, 'k: {}'.format(k), 'q/r: {}/{}'.format(d[-1], d[-1] * k)]), ax)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'top_k_hists_' + model + '.png'))


def hr_hists(qBX, qBY, rBX, rBY, k=50, model=''):
    print('Building Hamming radius histograms')

    def hr_hist_data(qB, rB, k):
        n = len(qB)
        dicts = []
        max_hr = 0
        for i in range(n):

            ham_dist = calc_hamming_dist(qB[i, :], rB).squeeze().detach().cpu()
            ham_dist_sorted, idxs = torch.sort(ham_dist)

            ham_dist_sorted_k = ham_dist_sorted[:k].cpu().numpy()
            values, counts = np.unique(ham_dist_sorted_k, return_counts=True)

            temp_d = {}

            for v, c in zip(values, counts):
                temp_d[int(v)] = c
                if v > max_hr:
                    max_hr = int(v)
            dicts.append(temp_d)

        hr_list = []
        for hr in range(max_hr + 1):
            hr_list_temp = []
            for d in dicts:
                if hr in d.keys():
                    hr_list_temp.append(d[hr])
                else:
                    hr_list_temp.append(0)
            hr_list.append(np.array(hr_list_temp))

        return hr_list

    def plot_hr_hist(ds, tag, ax):
        bars = range(len(ds[0]))
        offsets = np.zeros(len(ds[0]))
        for i, d in enumerate(ds):
            ax.bar(bars, d, bottom=offsets, width=1, label=str(i))
            offsets = offsets + d
        plt.title(tag.upper(), size=20, weight='medium')
        ax.legend(title='HR', bbox_to_anchor=(1, 1), loc='upper left')
        ax.set_xlabel('samples')
        ax.set_ylabel('K')
        plt.grid(axis='y')

    i2t = hr_hist_data(qBX, rBY, k)
    t2i = hr_hist_data(qBY, rBX, k)
    i2i = hr_hist_data(qBX, rBX, k)
    t2t = hr_hist_data(qBY, rBY, k)

    data = [i2t, t2i, i2i, t2t]
    tags = ['i2t', 't2i', 'i2i', 't2t']

    fig = plt.figure(figsize=(16, 8))
    for i, (tag, d) in enumerate(zip(tags, data)):
        ax = fig.add_subplot(2, 2, i + 1)
        plot_hr_hist(d, ', '.join([tag, model, 'k: {}'.format(k)]), ax)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'hr_hists_' + model + '.png'))


def bucket_hists(qBX, qBY, rBX, rBY, model, maps_r0, hash_dim):
    print('Building bucket histograms')

    def get_dict_of_binaries(binary_codes):

        def bin2dec(bin):
            bin = bin.detach().cpu().numpy()
            dec = np.uint64(0)
            for mag, bit in enumerate(bin[::-1]):
                dec += np.uint64(1 if bit >= 0 else 0) * np.power(np.uint64(2), np.uint64(mag), dtype=np.uint64)
            return dec

        dict_of_binaries = {}
        l = binary_codes.shape[0]
        for i in range(l):
            dec = bin2dec(binary_codes[i])
            if dec not in dict_of_binaries:
                dict_of_binaries[dec] = 1
            else:
                dict_of_binaries[dec] += 1

        return dict_of_binaries

    def get_stacked_bar_dict(qBd, rBd):
        joint_dict = qBd.copy()
        for k, v in joint_dict.items():
            joint_dict[k] = (v, 0)
        for k, v in rBd.items():
            if k in joint_dict:
                joint_dict[k] = (joint_dict[k][0], v)
            else:
                joint_dict[k] = (0, v)
        return joint_dict

    def plot_stacked_bar(stacked_bar_dict, tag, ax):
        labels = [str(i) for i in stacked_bar_dict.keys()]
        q = [i[0] for i in stacked_bar_dict.values()]
        r = [i[1] for i in stacked_bar_dict.values()]
        width = 1

        ax.bar(labels, q, width, label='Query')
        ax.bar(labels, r, width, bottom=q, label='Recall')
        plt.xticks([])
        plt.grid()
        ax.set_ylabel('Quantity')
        ax.set_title(tag.upper(), size=50, weight='medium')
        ax.legend()
        plt.tight_layout()

    qBXd = get_dict_of_binaries(qBX)
    qBYd = get_dict_of_binaries(qBY)
    rBXd = get_dict_of_binaries(rBX)
    rBYd = get_dict_of_binaries(rBY)

    i2t = get_stacked_bar_dict(qBXd, rBYd)
    t2i = get_stacked_bar_dict(qBYd, rBXd)
    i2i = get_stacked_bar_dict(qBXd, rBXd)
    t2t = get_stacked_bar_dict(qBYd, rBYd)

    tags = ['i2t', 't2i', 'i2i', 't2t']
    dicts = [i2t, t2i, i2i, t2t]

    fig = plt.figure(figsize=(60, 40))
    for i, (tag, d, mr0) in enumerate(zip(tags, dicts, maps_r0)):
        bins_used = 'buckets: {} / {}'.format(len(d), 2 ** hash_dim)
        experiment = ', '.join([tag, model, "mAP HR0: {:3.3f}".format(mr0), bins_used])
        ax = fig.add_subplot(2, 2, i + 1)
        plot_stacked_bar(d, experiment, ax)
    plt.savefig(os.path.join('plots', 'bucket_hists_' + model + '.png'))
