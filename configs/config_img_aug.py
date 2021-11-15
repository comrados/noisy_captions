import argparse
from .base_config import BaseConfig

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', default='rsicd', help='ucm or rsicd', type=str)
parser.add_argument('--img-aug', default='random_crop_only', type=str,
                    help="image transform set: see 'image_aug_transform_sets' variable")
parser.add_argument('--rot-deg', default=[-10, -5], nargs=2, type=int,
                    help="random rotation degrees range (2 integers) for 'rotation_cc': min_deg, max_deg")
parser.add_argument('--blur-val', default=[3., 3., 1.1, 1.3], nargs=4, type=float,
                    help="gaussian blur parameters for 'blur_cc': kernel_w, kernel_h, sigma_min, sigma_max")
parser.add_argument('--jit-str', default=[0.5], nargs=1, type=float,
                    help="color jitter strength 'jitter_cc': jitter strength")

args = parser.parse_args()

dataset = args.dataset
img_aug = args.img_aug

ard = args.rot_deg
abv = args.blur_val
js = args.jit_str


class ConfigImgPrep(BaseConfig):
    if dataset == 'ucm':
        image_emb_batch_size = 525  # 1/4 of dataset
        image_folder_preload = False  # Load all images before running (will be loading on request otherwise)
        image_aug_number = 1  # number generated augmented images

    if dataset == 'rsicd':
        image_emb_batch_size = 500
        image_folder_preload = False  # Load all images before running (will be loading on request otherwise)
        image_aug_number = 1  # number generated augmented images

    img_aug_set = img_aug  # name of image transform set to select, check image_aug_transform_sets
    # transforms will be applied in the dictionary order
    # 'blur' gaussian blur with given kernel size and sigma range
    # 'rotation' random rotation in given ranges of degrees
    # 'affine' random affine transform in given ranges of degrees
    # 'center_crop' center crop with given dimensions
    # 'random_crop' random crop with given dimensions
    # 'jitter' color jittering strength
    # 'each_img_random' - random transform from list for each image
    image_aug_transform_sets = {
        'center_crop_only': {'center_crop': (200, 200)},

        'random_crop_only': {'random_crop': (200, 200)},

        'aug_center': {'blur': ((3, 3), (0.1, 0.3)),
                       'rotation': [(5, 10), (-10, -5)],
                       # 'affine': [(5, 10), (-10, -5)],
                       'center_crop': (200, 200)},

        'aug_random': {'blur': ((3, 3), (0.1, 0.3)),
                       'rotation': [(5, 10), (-10, -5)],
                       # 'affine': [(5, 10), (-10, -5)],
                       'random_crop': (200, 200)},

        'rotation_cc': {'rotation': [tuple(ard)],
                        'center_crop': (200, 200)},

        'blur_cc': {'blur': (tuple([int(i) for i in abv[:2]]), tuple(abv[2:])),
                    'center_crop': (200, 200)},

        'jitter_cc': {'jitter': js[0],
                      'center_crop': (200, 200)},

        'each_img_random': ['rotation_cc', 'blur_cc', 'jitter_cc']
    }

    if img_aug_set == 'rotation_cc':
        image_emb_aug_file = "./data/image_emb_{}_aug_{}_{}_{}.h5".format(dataset.upper(), img_aug_set, ard[0], ard[1])
    elif img_aug_set == 'blur_cc':
        image_emb_aug_file = "./data/image_emb_{}_aug_{}_{}_{}.h5".format(dataset.upper(), img_aug_set,
                                                                          '_'.join([str(int(i)) for i in abv[:2]]),
                                                                          '_'.join([str(i) for i in abv[2:]]))
    elif img_aug_set == 'jitter_cc':
        image_emb_aug_file = "./data/image_emb_{}_aug_{}_{}.h5".format(dataset.upper(), img_aug_set, js[0])
    else:
        image_emb_aug_file = "./data/image_emb_{}_aug_{}.h5".format(dataset.upper(), img_aug_set)
    image_emb_file = "./data/image_emb_{}_aug.h5".format(dataset.upper())

    def __init__(self, args):
        super(ConfigImgPrep, self).__init__(args)


cfg = ConfigImgPrep(args)
