import argparse
from .base_config import BaseConfig

parser = argparse.ArgumentParser(description='')
parser.add_argument('--test', default=False, help='train or test', action='store_true')
parser.add_argument('--bit', default=64, help='hash bit', type=int)
parser.add_argument('--model', default='UNHD', help='model type', type=str)
parser.add_argument('--epochs', default=150, help='training epochs', type=int)
parser.add_argument('--tag', default='test', help='model tag', type=str)
parser.add_argument('--dataset', default='ucm', help='ucm or rsicd', type=str)
parser.add_argument('--preset', default='clean', help='data presets, see available in config.py', type=str)
parser.add_argument('--alpha', default=0, help='alpha hyperparameter (La)', type=float)
parser.add_argument('--beta', default=0.001, help='beta hyperparameter (Lq)', type=float)
parser.add_argument('--gamma', default=0, help='gamma hyperparameter (Lbb)', type=float)
parser.add_argument('--contrastive-weights', default=[1.0, 0.0, 0.0], type=float, nargs=3,
                    help='contrastive loss component weights: [inter, intra_img, intra_txt]')

parser.add_argument('--img-aug-emb', default=None, type=str, help='overrides augmented image embeddings file (u-curve)')
parser.add_argument('--txt-aug-emb', default=None, type=str, help='overrides augmented text embeddings file (noise)')

parser.add_argument('--noise-wrong-caption', default=.5, type=float, help="probability of 'wrong caption' noise")
parser.add_argument('--clean-captions', default=.2, type=float, help="amount of free captions in dataset")
parser.add_argument('--noise-weights', default='normal', type=str, choices=['normal', 'exp', 'dis', 'ones'], help="amount of free captions in dataset")
parser.add_argument('--clean-epochs', default=75, help='number of clean epochs', type=int)

args = parser.parse_args()

dataset = args.dataset
preset = args.preset

alpha = args.alpha
beta = args.beta
gamma = args.gamma
contrastive_weights = args.contrastive_weights
wrong_noise_caption_prob = args.noise_wrong_caption
clean_captions = args.clean_captions
noise_weights = args.noise_weights
clean_epochs = args.clean_epochs


class ConfigModel(BaseConfig):
    preset = preset.lower()

    if preset == 'clean':
        # default for texts
        image_emb_for_model = "./data/image_emb_{}_aug_center_crop_only.h5".format(dataset.upper())
        caption_emb_for_model = "./data/caption_emb_{}_aug.h5".format(dataset.upper())

        image_emb_aug_for_model = "./data/image_emb_{}_aug_aug_center.h5".format(dataset.upper())
        caption_emb_aug_for_model = "./data/caption_emb_{}_aug.h5".format(dataset.upper())

        dataset_json_for_model = "./data/augmented_{}.json".format(dataset.upper())
    else:
        raise Exception('Nonexistent preset: {}'.format(preset))

    if args.img_aug_emb is not None:
        image_emb_aug_for_model = args.img_aug_emb

    if args.txt_aug_emb is not None:
        caption_emb_aug_for_model = args.txt_aug_emb

    if dataset == 'ucm':
        label_dim = 21
        # dataset settings
        dataset_file = "../data/dataset_UCM_aug_captions_images.h5"  # Resulting dataset file
        dataset_train_split = 0.5  # part of all data, that will be used for training
        # (1 - dataset_train_split) - evaluation data
        dataset_query_split = 0.2  # part of evaluation data, that will be used for query
        # (1 - dataset_train_split) * (1 - dataset_query_split) - retrieval data

    if dataset == 'rsicd':
        label_dim = 31
        # dataset settings
        dataset_file = "../data/dataset_RSICD_aug_captions_images.h5"  # Resulting dataset file
        dataset_train_split = 0.5  # part of all data, that will be used for training
        # (1 - dataset_train_split) - evaluation data
        dataset_query_split = 0.2  # part of evaluation data, that will be used for query
        # (1 - dataset_train_split) * (1 - dataset_query_split) - retrieval data

    build_plots = False

    wrong_noise_caption_prob = wrong_noise_caption_prob
    clean_captions = clean_captions
    noise_weights = noise_weights

    model_type = 'UNHD'
    batch_size = 256
    image_dim = 512
    text_dim = 768
    hidden_dim = 1024 * 4
    hash_dim = 128
    noise_dim = image_dim + text_dim

    lr = 0.0001
    clean_epochs = clean_epochs
    max_epoch = 100
    valid = True  # validation
    valid_freq = 150  # validation frequency (epochs)
    alpha = alpha  # adv loss
    beta = beta  # quant loss
    gamma = gamma  # bb loss
    contrastive_weights = contrastive_weights  # [inter, intra_img, intra_txt]

    retrieval_map_k = 5

    tag = 'test'

    def __init__(self, args):
        super(ConfigModel, self).__init__(args)
        self.test = args.test
        self.hash_dim = args.bit
        self.model_type = args.model
        self.max_epoch = args.epochs
        self.tag = args.tag


cfg = ConfigModel(args)
