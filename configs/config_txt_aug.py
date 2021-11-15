import argparse
from .base_config import BaseConfig

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', default='rsicd', help='ucm or rsicd', type=str)
parser.add_argument('--txt-aug', default='backtranslation-prob', type=str,
                    help="image transform set: 'rule-based', 'backtranslation-prob', 'backtranslation-chain'")

args = parser.parse_args()

dataset = args.dataset
txt_aug = args.txt_aug


class ConfigTxtPrep(BaseConfig):
    if dataset == 'ucm':
        caption_token_length = 64  # use hardcoded number, max token length for clean data is 26

    if dataset == 'rsicd':
        caption_token_length = 64  # use hardcoded number, max token length for clean data is 40

    caption_hidden_states = 4  # Number of last BERT's hidden states to use
    caption_hidden_states_operator = 'sum'  # "How to combine hidden states: 'sum' or 'concat'"

    caption_aug_rb_glove_sim_threshold = 0.65
    caption_aug_rb_bert_score_threshold = 0.75

    # caption_aug_type - augmentation method 'prob' or 'chain'
    # caption_aug_method:
    #   'prob' - sentence augmented via random lang with prob proportional to weight from captions_aug_bt_lang_weights
    #   'chain' - translates in chain en -> lang1 -> lang2 -> ... -> en (weight values are ignored)
    if txt_aug.startswith('rule'):
        caption_aug_type = 'rule-based'
        caption_aug_method = None
    else:
        caption_aug_type = 'backtranslation'
        if txt_aug.endswith('prob'):
            caption_aug_method = 'prob'
        else:
            caption_aug_method = 'chain'

    # en -> random language -> en, lang_prob = lang_weight / sum(lang_weights)
    caption_aug_bt_lang_weights = {'es': 1, 'de': 1}
    # caption_aug_bt_lang_weights = {'ru': 1, 'bg': 1}

    caption_aug_dataset_json = "./data/augmented_{}.json".format(dataset.upper())
    caption_emb_file = "./data/caption_emb_{}_aug.h5".format(dataset.upper())
    caption_emb_aug_file_rb = "./data/caption_emb_{}_aug_rb.h5".format(dataset.upper())
    caption_emb_aug_file_bt_prob = "./data/caption_emb_{}_aug_bt_prob.h5".format(dataset.upper())
    caption_emb_aug_file_bt_chain = "./data/caption_emb_{}_aug_bt_chain.h5".format(dataset.upper())

    noisy_dataset_json = "./data/noisy_UCM.json"
    clean_dataset_reformat = True  # reformat text of 'dataset_json_file' and remove sentence tokens (if exist)
    noise_chance = 0.5  # overall noise insertion chance
    caption_replace_chance = 0.05  # a chance (among all noise cases) caption will be replaced with random one
    # dictionary of int, the higher rate, the more often noise type is selected
    noise_rates = {'char_missing': 1, 'char_swap': 1, 'char_replace': 1, 'char_duplicate': 1, 'token_remove': 1}
    # dictionary of (int, int), min and max number of occurrences per caption; actual number can be in range(min, max+1)
    noise_cases = {'char_missing': (2, 5), 'char_swap': (2, 5), 'char_replace': (2, 5), 'char_duplicate': (2, 4),
                   'token_remove': (1, 2)}

    def __init__(self, args):
        super(ConfigTxtPrep, self).__init__(args)


cfg = ConfigTxtPrep(args)
