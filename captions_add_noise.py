from utils import read_json, write_json
from configs.config_txt_aug import cfg

import random
import copy


def remove_tokens(data):
    """
    Removes 'tokens' key from caption record, if exists; halves the size of the file

    :param data: original data
    :return: data without tokens
    """
    for img in data['images']:
        for sent in img['sentences']:
            try:
                sent.pop("tokens")
            except:
                pass
    return data


class NoiseCreator:

    def __init__(self, cfg):
        self.cfg = cfg

        self.noise_type_dict = {0: 'none', 1: 'caption replace', 2: 'missing char', 3: 'char swap',
                                4: 'char duplication', 5: 'char replacement', 6: 'removed token'}

        self.noise_functions = {'char_missing': self.remove_char, 'char_swap': self.swap_char,
                                'char_replace': self.replace_char, 'char_duplicate': self.duplicate_char,
                                'token_remove': self.remove_token}

        self.noise_rates_list = self.make_noise_rates_list()

    @staticmethod
    def get_captions_dict(data):
        """
        Get dict of captions

        :param data: original data from JSON
        :return: dict of strings (captions)
        """

        captions = {}
        for img in data['images']:
            for sent in img['sentences']:
                captions[sent['sentid']] = sent['raw']
        print("Total number of captions:", len(captions))
        return captions

    def make_noise_rates_list(self):
        nr_list = []
        for k, v in self.cfg.noise_rates.items():
            nr_list += [k] * v
        return nr_list

    @staticmethod
    def remove_char(sent, times):
        # remove random character
        for i in range(times):
            n = random.randint(0, len(sent) - 2)
            sent = sent[:n] + sent[n+1:]
        return sent, 2

    @staticmethod
    def swap_char(sent, times):

        def swap(s, i, j):
            l = list(s)
            l[i], l[j] = l[j], l[i]
            return ''.join(l)

        for i in range(times):
            n = random.randint(0, len(sent) - 2)
            sent = swap(sent, n, n+1)

        # swap random characters
        return sent, 3

    @staticmethod
    def duplicate_char(sent, times):
        # duplicate random character
        for i in range(times):
            n = random.randint(0, len(sent) - 2)
            sent = sent[:n] + sent[n] + sent[n:]
        return sent, 4

    @staticmethod
    def replace_char(sent, times):

        char_choice = [32] + [i for i in range(97, 123)]  # ' ' + a-z

        for i in range(times):
            n = random.randint(0, len(sent) - 2)
            c = random.choice(char_choice)
            sent = sent[:n] + chr(c) + sent[n + 1:]
        return sent, 5

    @staticmethod
    def remove_token(sent, times):
        # remove token
        last = None
        tokens = sent.split(' ')
        if tokens[-1] in ['.', ',', '?', '!']:
            last = tokens[-1]
            tokens.remove(tokens[-1])

        for i in range(times):
            n = random.randint(0, len(tokens) - 1)
            tokens.remove(tokens[n])

        new_sent = ' '.join(tokens)
        if last:
            new_sent = new_sent + ' ' + last

        return new_sent, 6

    def add_random_noise(self, sent):
        noise_choice = random.choice(self.noise_rates_list)
        noise_function = self.noise_functions[noise_choice]
        times = random.randint(self.cfg.noise_cases[noise_choice][0], self.cfg.noise_cases[noise_choice][1])
        noisy_caption, noise_id = noise_function(sent, times)
        return noisy_caption, noise_id, times

    @staticmethod
    def roll(threshold):
        roll_outcome = random.random()
        return roll_outcome < threshold  # True if rolled value is lower than threshold

    def add_noise_to_data(self, clean_data):
        random.seed(cfg.seed)

        noisy_data = copy.deepcopy(clean_data)

        cap_dict = self.get_captions_dict(noisy_data)

        for img in noisy_data['images']:
            for sent in img['sentences']:
                if self.roll(self.cfg.noise_chance):
                    if self.roll(self.cfg.caption_replace_chance):
                        replacement_id = random.randint(0, len(cap_dict) - 1)
                        sent['raw'] = cap_dict[replacement_id]
                        sent['noisetype'] = self.noise_type_dict[1]
                        sent['noisecode'] = 1
                        sent['noisetimes'] = 1
                        continue
                    else:
                        noisy_caption, noise_id, noise_times = self.add_random_noise(sent['raw'])
                        sent['raw'] = noisy_caption
                        sent['noisetype'] = self.noise_type_dict[noise_id]
                        sent['noisecode'] = noise_id
                        sent['noisetimes'] = noise_times
                else:
                    sent['noisetype'] = self.noise_type_dict[0]
                    sent['noisecode'] = 0
                    sent['noisetimes'] = 0

        return noisy_data


if __name__ == '__main__':
    print("ADD NOISE TO CAPTIONS")

    # read original captions (clean data)
    clean_data = read_json(cfg.dataset_json_file)

    nc = NoiseCreator(cfg)

    # add noise to captions
    noisy_data = nc.add_noise_to_data(clean_data)

    # output noisy captions
    write_json(cfg.noisy_dataset_json, remove_tokens(noisy_data))

    # reformat text in original JSON file (only for appearance enchantment)
    if cfg.clean_dataset_reformat:
        write_json(cfg.dataset_json_file, remove_tokens(clean_data))
    print("DONE\n\n\n")
