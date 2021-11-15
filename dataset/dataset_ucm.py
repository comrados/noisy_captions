import torch
import numpy as np
import random
from configs.config import cfg

from utils import select_idxs, select_k_idxs_from_pop


class AbstractDataset(torch.utils.data.Dataset):

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        self.seed = seed
        self.image_replication_factor = 1  # default value, how many times we need to replicate image

        self.images = images
        self.captions = captions
        self.labels = labels

        self.captions_aug = captions_aug
        self.images_aug = images_aug

        self.idxs = np.array(idxs[0])
        self.idxs_cap = np.array(idxs[1])

    def __getitem__(self, index):
        return

    def __len__(self):
        return


class DatasetQuadrupletAugmentedTxtImg(AbstractDataset):
    """
    Class for dataset representation.

    Quadruplet dataset sample - img-img-txt-txt
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        caption_idxs = select_idxs(len(self.captions), 1, 5, seed=self.seed)[0]
        self.captions = self.captions[caption_idxs]
        self.captions_aug = self.captions_aug[caption_idxs]
        self.idxs_cap = self.idxs_cap[caption_idxs]

    def __getitem__(self, index):
        """
        Returns a tuple (img1, img2, txt1, txt2, label)

        :param index: index of sample
        :return: tuple (img1, img2, txt1, txt2, label)
        """
        return (
            index,
            (self.idxs[index], self.idxs[index], self.idxs_cap[index], self.idxs_cap[index]),
            torch.from_numpy(self.images[index].astype('float32')),
            torch.from_numpy(self.images_aug[index].astype('float32')),
            torch.from_numpy(self.captions[index].astype('float32')),
            torch.from_numpy(self.captions_aug[index].astype('float32')),
            self.labels[index]
        )

    def __len__(self):
        return len(self.images)


class DatasetDuplet1(AbstractDataset):
    """
    Class for dataset representation.

    Each image has 5 corresponding captions

    Duplet dataset sample - img-txt (image and corresponding caption)
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization.

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        caption_idxs = select_idxs(len(self.captions), 1, 5, seed=self.seed)[0]
        self.captions = self.captions[caption_idxs]

    def __getitem__(self, index):
        """
        Returns a tuple (img, txt, label) - image and corresponding caption

        :param index: index of sample
        :return: tuple (img, txt, label)
        """
        return (
            index,
            (self.idxs[index], self.idxs_cap[index]),
            torch.from_numpy(self.images[index].astype('float32')),
            torch.from_numpy(self.captions[index].astype('float32')),
            self.labels[index]
        )

    def __len__(self):
        return len(self.images)


class DatasetQuadrupletAugmentedTxtImgNoiseWrongCaption(AbstractDataset):
    """
    Class for dataset representation.

    Quadruplet dataset sample - img-img-txt-txt
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        caption_idxs = select_idxs(len(self.captions), 1, 5, seed=self.seed)[0]
        self.captions = self.captions[caption_idxs]
        self.captions_aug = self.captions_aug[caption_idxs]
        self.idxs_cap = self.idxs_cap[caption_idxs]
        self.counter = 0

        self.wrong_caption_idxs = self.get_wrong_caption_indexes()
        self.clean_idxs = select_k_idxs_from_pop(int(len(self.captions) * 0.2), self.captions, seed=self.seed)

        self.print_counter()

    def __getitem__(self, index):
        """
        Returns a tuple (img1, img2, txt1, txt2, label)

        :param index: index of sample
        :return: tuple (img1, img2, txt1, txt2, label)
        """
        if index in self.clean_idxs:
            idx_wrong = index
        else:
            idx_wrong = self.wrong_caption_idxs[index]
        return (
            index,
            (self.idxs[index], self.idxs[index], self.idxs_cap[index], self.idxs_cap[idx_wrong]),
            torch.from_numpy(self.images[index].astype('float32')),
            torch.from_numpy(self.images_aug[index].astype('float32')),
            torch.from_numpy(self.captions[idx_wrong].astype('float32')),
            torch.from_numpy(self.captions_aug[idx_wrong].astype('float32')),
            self.labels[index]
        )

    def generate_wrong_caption_index(self, index):
        true_lab = self.labels[index]

        while True:
            new_idx = random.choice(range(len(self.images)))
            new_lab = self.labels[new_idx]
            if true_lab != new_lab:
                break

        return new_idx

    def get_wrong_caption_indexes(self):
        idxs = []
        for i in range(len(self.images)):
            roll = random.random()
            if roll < cfg.wrong_noise_caption_prob:
                idxs.append(self.generate_wrong_caption_index(i))
                self.counter += 1
            else:
                idxs.append(i)
        return idxs

    def print_counter(self):
        print("Replaced captions: {}/{}".format(self.counter, len(self.images)))

    def __len__(self):
        return len(self.images)


class DatasetQuadrupletAugmentedTxtImgNoiseWrongCaptionClean(AbstractDataset):
    """
    Class for dataset representation.

    Quadruplet dataset sample - img-img-txt-txt
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        caption_idxs = select_idxs(len(self.captions), 1, 5, seed=self.seed)[0]
        self.captions = self.captions[caption_idxs]
        self.captions_aug = self.captions_aug[caption_idxs]
        self.idxs_cap = self.idxs_cap[caption_idxs]
        self.counter = 0

        self.clean_idxs = select_k_idxs_from_pop(int(len(self.captions) * 0.2), self.captions, seed=self.seed)

    def __getitem__(self, index):
        """
        Returns a tuple (img1, img2, txt1, txt2, label)

        :param index: index of sample
        :return: tuple (img1, img2, txt1, txt2, label)
        """
        idx_clean = self.clean_idxs[index]
        return (
            index,
            (self.idxs[idx_clean], self.idxs[idx_clean], self.idxs_cap[idx_clean], self.idxs_cap[idx_clean]),
            torch.from_numpy(self.images[idx_clean].astype('float32')),
            torch.from_numpy(self.images_aug[idx_clean].astype('float32')),
            torch.from_numpy(self.captions[idx_clean].astype('float32')),
            torch.from_numpy(self.captions_aug[idx_clean].astype('float32')),
            self.labels[index]
        )

    def __len__(self):
        return len(self.clean_idxs)
