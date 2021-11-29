from models.UHND import UNHD
from loss.contrastive_loss import NTXentLoss
from utils import write_pickle, calc_map_k, calc_map_rad, pr_curve, p_top_k
from utils import retrieval2png, bucket_hists, top_k_hists, hr_hists
from torch import autograd
import torch
from torch.nn.functional import one_hot
from torch.optim import Adam, SGD, AdamW
import random

import time
import os


class ControllerUNHD:
    """
    Training/evaluation controller.
    """

    def __init__(self, log, cfg, dataloaders):
        self.since = time.time()
        self.logger = log
        self.cfg = cfg
        self.path = 'checkpoints/' + '_'.join([self.cfg.dataset, str(self.cfg.hash_dim), self.cfg.tag, str(self.cfg.wrong_noise_caption_prob), str(self.cfg.clean_captions), self.cfg.noise_weights])

        self.device = torch.device(self.cfg.cuda_device if torch.cuda.is_available() else "cpu")

        self.dataloader_quadruplets, self.dataloader_quadruplets_clean, self.dataloader_q, self.dataloader_db = dataloaders

        self.B, self.Hi1, self.Hi2, self.Ht1, self.Ht2 = self.init_hashes()

        self.losses = []
        self.probabs = []

        self.maps_max = {'i2t': 0., 't2i': 0., 'i2i': 0., 't2t': 0., 'avg': 0.}
        self.maps = {'i2t': [], 't2i': [], 'i2i': [], 't2t': [], 'avg': []}

        self.model = self.get_model()
        self.optimizer_gen, self.optimizer_dis = self.get_optimizers()
        self.contr_loss = NTXentLoss()

    def init_hashes(self):
        """
        Initialize hash values (either zeros or random, see below)

        :return: initialized hash values
        """
        dataset_size = len(self.dataloader_quadruplets.dataset)
        B = torch.randn(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        Hi1 = torch.zeros(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        Hi2 = torch.zeros(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        Ht1 = torch.zeros(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        Ht2 = torch.zeros(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        return B, Hi1, Hi2, Ht1, Ht2

    def get_model(self):
        """
        Initialize model

        :returns: instance of NN model
        """
        return UNHD(self.cfg.image_dim, self.cfg.text_dim, self.cfg.hidden_dim, self.cfg.hash_dim)

    def get_optimizers(self):
        """
        Initialize learning optimizers

        :returns: learning optimizers
        """
        optimizer_gen = Adam([
            {'params': self.model.image_module.parameters()},
            {'params': self.model.text_module.parameters()}
        ], lr=self.cfg.lr, weight_decay=0.0005)

        optimizer_dis = AdamW(self.model.noise_dis.parameters(), lr=0.0001, betas=(0.5, 0.9), weight_decay=0.001)
        # optimizer_dis = SGD(self.model.noise_dis.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)
        # optimizer_dis = Adam(self.model.noise_dis.parameters(), lr=0.0001, betas=(0.5, 0.9), weight_decay=0.001)

        return optimizer_gen, optimizer_dis

    def train_epoch(self, epoch):
        """
        Train model for 1 epoch

        :param: epoch: current epoch
        """
        t1 = time.time()

        self.model.to(self.device).train()

        e_losses_dict = {'adv': 0., 'bb': 0., 'quant': 0., 'ntxent': 0., 'sum': 0.}

        weights_dict = {'w1_clean': None, 'w2_clean': None, 'w1_noise': None, 'w2_noise': None}
        weights_meta_dict = {'w1_clean': None, 'w2_clean': None, 'w1_noise': None, 'w2_noise': None}

        for i, (dl_idx, sample_idxs, img1, img2, txt1, txt2, label, noise, clean) in enumerate(self.dataloader_quadruplets):
            img1 = img1.to(self.device)  # original images
            img2 = img2.to(self.device)  # augmented images
            txt1 = txt1.to(self.device)  # original texts
            txt2 = txt1.to(self.device)  # augmented texts
            h_img1, h_img2, h_txt1, h_txt2 = self.model(img1, img2, txt1, txt2)

            self.Hi1[dl_idx, :] = h_img1
            self.Hi2[dl_idx, :] = h_img2
            self.Ht1[dl_idx, :] = h_txt1
            self.Ht2[dl_idx, :] = h_txt2

            pairs1 = self.generate_pairs(img1.clone().detach(), txt1.clone().detach())
            pairs2 = self.generate_pairs(img2.clone().detach(), txt2.clone().detach())

            noise_weights_dict = self.generate_pair_weights(pairs1, pairs2)

            w1, w2 = noise_weights_dict[self.cfg.noise_weights]

            weights_dict = self.update_weights_dict(weights_dict, weights_meta_dict, w1, w2, noise, clean)

            e_losses_dict = self.train_hash_network(h_img1, h_img2, h_txt1, h_txt2, dl_idx, e_losses_dict, weights=w1)

        self.losses.append(e_losses_dict)

        self.B = (((self.Hi1.detach() + self.Hi2.detach()) / 2 +
                   (self.Ht1.detach() + self.Ht2.detach()) / 2) / 2).sign()

        self.update_optimizer_params(epoch)

        self.density_plot(epoch, weights_dict)

        delta_t = time.time() - t1
        s = '[{}/{}] Train: {:.3f}s, Losses: Q: {quant:10.2f}, NTX: {ntxent:10.2f}'
        self.logger.info(s.format(epoch + 1, self.cfg.max_epoch, delta_t, **e_losses_dict))

    def density_plot(self, epoch, weights_dict):
        """
        build weights distribution density plot

        :param epoch: epoch
        :param weights_dict:  dictionary
        :return:
        """
        if (epoch + 1) == self.cfg.clean_epochs + 3:

            import seaborn as sns
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(6, 6))

            ax = plt.subplot(1, 1, 1)

            sns.distplot(weights_dict['w1_clean'], bins=25, label='clean_captions', ax=ax)
            sns.distplot(weights_dict['w2_clean'], bins=25, label='clean_aug_captions', ax=ax)
            sns.distplot(weights_dict['w1_noise'], bins=25, label='noisy_captions', ax=ax)
            sns.distplot(weights_dict['w2_noise'], bins=25, label='noisy_aug_captions', ax=ax)

            ax.set_xlabel('probability')
            ax.set_ylabel('frequency')

            plt.legend()
            noise = str(self.cfg.wrong_noise_caption_prob)
            clean = str(self.dataloader_quadruplets_clean.dataset.clean_captions)
            plt.title(' '.join([self.cfg.dataset, 'noise level:', noise, 'clean sapmles: ', clean]))
            plt.tight_layout()

            print(os.path.join('plots', 'noise_distplot_{}_{}_{}_{}.png'.format(self.cfg.dataset, noise, clean, self.cfg.noise_weights)))
            plt.savefig(os.path.join('plots', 'noise_distplot_{}_{}_{}_{}.png'.format(self.cfg.dataset, noise, clean, self.cfg.noise_weights)))
            plt.savefig(os.path.join('plots', 'noise_distplot_{}_{}_{}_{}.pdf'.format(self.cfg.dataset, noise, clean, self.cfg.noise_weights)))

    @staticmethod
    def update_weights_dict(weights_dict, weights_meta_dict, w1, w2, noise, meta):
        """
        put new values or initialize weights dict

        :param weights_dict:
        :param weights_meta_dict:
        :param w1:
        :param w2:
        :param noise:
        :param meta:
        :return:
        """

        mask_noise = noise.eq(1)
        mask_clean = noise.eq(0)
        w1_clean = torch.masked_select(w1, mask_clean).detach()
        w2_clean = torch.masked_select(w2, mask_clean).detach()

        w1_noise = torch.masked_select(w1, mask_noise).detach()
        w2_noise = torch.masked_select(w2, mask_noise).detach()

        if weights_dict['w1_clean'] is None:
            weights_dict['w1_clean'] = w1_clean
            weights_dict['w2_clean'] = w2_clean
            weights_dict['w1_noise'] = w1_noise
            weights_dict['w2_noise'] = w2_noise
        else:
            weights_dict['w1_clean'] = torch.hstack([weights_dict['w1_clean'], w1_clean])
            weights_dict['w2_clean'] = torch.hstack([weights_dict['w2_clean'], w2_clean])
            weights_dict['w1_noise'] = torch.hstack([weights_dict['w1_noise'], w1_noise])
            weights_dict['w2_noise'] = torch.hstack([weights_dict['w2_noise'], w2_noise])

        mask_meta1 = meta.eq(1)
        mask_meta0 = meta.eq(0)

        w1_meta1 = torch.masked_select(w1, mask_meta1).detach()
        w1_meta0 = torch.masked_select(w1, mask_meta0).detach()

        noise_meta1 = torch.masked_select(noise, mask_meta1).detach()
        noise_meta0 = torch.masked_select(noise, mask_meta0).detach()

        return weights_dict

    def generate_pair_weights(self, *pairs):
        """
        generate weights for feature pairs

        :param pairs: img-txt joint-pairs
        :return:
        """

        output = []
        for pair in pairs:
            output.append(torch.sigmoid(self.model.discriminate_noise(pair)).detach().cpu())

        # exponentiated weights
        output_pow = [torch.abs(torch.pow(x, 3)) for x in output]

        # discrete weights
        try:
            threshold = (self.probabs[-1]['r1'] + self.probabs[-1]['f1']) / 2
        except:
            threshold = 0.5
        output_dis = [x.gt(threshold).type(torch.float) for x in output]

        # 1 weights
        output_ones = [torch.ones_like(x) for x in output]

        return {'normal': output, 'exp': output_pow, 'dis': output_dis, 'ones': output_ones}

    def train_epoch_clean_data(self, epoch):
        """
        Train model for 1 epoch with only clean data

        :param: epoch: current epoch
        """
        t1 = time.time()

        self.model.to(self.device).train()

        e_losses_dict = {'adv': 0., 'bb': 0., 'quant': 0., 'ntxent': 0., 'sum': 0.}

        e_probabs_dict = {'r1': None, 'r2': None, 'f1': None, 'f2': None}

        for i, (dl_idx, sample_idxs, img1, img2, txt1, txt2, label, noise, clean) in enumerate(self.dataloader_quadruplets_clean):
            img1 = img1.to(self.device)  # original images
            img2 = img2.to(self.device)  # augmented images
            txt1 = txt1.to(self.device)  # original texts
            txt2 = txt1.to(self.device)  # augmented texts
            h_img1, h_img2, h_txt1, h_txt2 = self.model(img1, img2, txt1, txt2)

            real1 = self.generate_pairs(img1.clone().detach(), txt1.clone().detach())
            real2 = self.generate_pairs(img2.clone().detach(), txt2.clone().detach())
            fake1 = self.generate_fake_pairs(img1.clone().detach(), txt1.clone().detach(), label.clone().detach())
            fake2 = self.generate_fake_pairs(img2.clone().detach(), txt2.clone().detach(), label.clone().detach())

            # intermixed augmented and non-augmented feature representations
            real3 = self.generate_pairs(img1.clone().detach(), txt2.clone().detach())
            real4 = self.generate_pairs(img2.clone().detach(), txt1.clone().detach())
            fake3 = self.generate_fake_pairs(img1.clone().detach(), txt2.clone().detach(), label.clone().detach())
            fake4 = self.generate_fake_pairs(img2.clone().detach(), txt1.clone().detach(), label.clone().detach())

            reals = [real1, real2, real3, real4]
            fakes = [fake1, fake2, fake3, fake4]
            #reals = [real1, real2]
            #fakes = [fake1, fake2]

            self.Hi1[dl_idx, :] = h_img1
            self.Hi2[dl_idx, :] = h_img2
            self.Ht1[dl_idx, :] = h_txt1
            self.Ht2[dl_idx, :] = h_txt2

            current_batch_size = len(dl_idx)
            self.train_discriminator(reals, fakes, current_batch_size)

            e_probabs_dict = self.update_e_probabs_dict(e_probabs_dict, real1, real2, fake1, fake2)
            e_losses_dict = self.train_hash_network(h_img1, h_img2, h_txt1, h_txt2, dl_idx, e_losses_dict)

        self.losses.append(e_losses_dict)
        self.probabs.append(self.calculate_mean_probabs(e_probabs_dict))

        self.B = (((self.Hi1.detach() + self.Hi2.detach()) / 2 +
                   (self.Ht1.detach() + self.Ht2.detach()) / 2) / 2).sign()

        self.update_optimizer_params(epoch)

        delta_t = time.time() - t1
        s = '[{}/{}] Train: {:.3f}s, Losses: Q: {quant:10.2f}, NTX: {ntxent:10.2f}'
        self.logger.info(s.format(epoch + 1, self.cfg.max_epoch, delta_t, **e_losses_dict))
        s2 = '[{}/{}]: Avg probabilities of real and fakes: R1: {r1:5.3f}, R2: {r2:5.3f}, F1: {f1:5.3f}, F2: {f2:5.3f}'
        self.logger.info(s2.format(epoch + 1, self.cfg.max_epoch, **self.calculate_mean_probabs(e_probabs_dict)))

    @staticmethod
    def generate_pairs(img, txt):
        """
        combine features into pairs (joint-pairs)

        :param img: feature img
        :param txt: feature txt
        :return:
        """
        return torch.cat((img, txt), 1)

    @staticmethod
    def generate_fake_pairs(img, txt, label):
        """
        generate fake img-txt feature pairs by combining non-corresponding img and txt features

        :param img: img features
        :param txt: txt features
        :param label: labels (for class checking)
        :return:
        """

        # same randomizer like in Dataset
        perm = []
        for i in range(len(txt)):
            flag = True
            while flag:
                new_idx = random.choice(range(len(txt)))
                new_lab = label[new_idx]
                true_lab = label[i]
                if true_lab != new_lab:
                    perm.append(new_idx)
                    flag = False
        return torch.cat((img, txt[perm]), 1)

    def update_e_probabs_dict(self, e_probabs_dict, real1, real2, fake1, fake2):
        """
        update or init probabilities (pair weights) dictionary for epoch

        :param e_probabs_dict:
        :param real1:
        :param real2:
        :param fake1:
        :param fake2:
        :return:
        """
        r1 = torch.sigmoid(self.model.discriminate_noise(real1)).detach().cpu()
        r2 = torch.sigmoid(self.model.discriminate_noise(real2)).detach().cpu()
        f1 = torch.sigmoid(self.model.discriminate_noise(fake1)).detach().cpu()
        f2 = torch.sigmoid(self.model.discriminate_noise(fake2)).detach().cpu()

        if e_probabs_dict['r1'] is None:
            e_probabs_dict['r1'] = r1
            e_probabs_dict['r2'] = r2
            e_probabs_dict['f1'] = f1
            e_probabs_dict['f2'] = f2
        else:
            e_probabs_dict['r1'] = torch.hstack([e_probabs_dict['r1'], r1])
            e_probabs_dict['r2'] = torch.hstack([e_probabs_dict['r2'], r2])
            e_probabs_dict['f1'] = torch.hstack([e_probabs_dict['f1'], f1])
            e_probabs_dict['f2'] = torch.hstack([e_probabs_dict['f2'], f2])

        return e_probabs_dict

    @staticmethod
    def calculate_mean_probabs(e_probabs_dict):
        """
        calculates mean probabilities from epoch prob dictionary

        :param e_probabs_dict:
        :return:
        """
        out = {'r1': None, 'r2': None, 'f1': None, 'f2': None}

        for k in out:
            out[k] = e_probabs_dict[k].mean().numpy()

        return out

    def train_discriminator(self, reals, fakes, batch_size):
        """
        Train hash discriminator

        :param: reals: list of real hashes batches
        :param: fakes: list of fake hashes batches
        :param: batch_size: current batch size
        """
        self.optimizer_dis.zero_grad()
        for f_real, f_fake in zip(reals, fakes):
            self.train_discriminator_step(f_real, f_fake, batch_size)
        self.optimizer_dis.step()

    def train_discriminator_step(self, v_real, v_fake, batch_size):
        """
        Train hash discriminator for one step

        :param: v_real: 'real' samples
        :param: v_fake: 'fake' samples
        :param: batch_size: current batch size
        """
        d_real = self.model.discriminate_noise(v_real.detach())
        d_real = -torch.log(torch.sigmoid(d_real)).mean()
        d_real.backward()

        # train with fake (TXT)
        d_fake = self.model.discriminate_noise(v_fake.detach())
        d_fake = -torch.log(torch.ones(batch_size).to(self.device) - torch.sigmoid(d_fake)).mean()
        d_fake.backward()

        # train with gradient penalty (GP)

        # interpolate real and fake data
        alpha = torch.rand(batch_size, self.cfg.noise_dim).to(self.device)
        interpolates = alpha * v_real.detach() + (1 - alpha) * v_fake.detach()
        interpolates.requires_grad_()
        disc_interpolates = self.model.discriminate_noise(interpolates)
        # get gradients with respect to inputs
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        # calculate penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # 10 is GP hyperparameter
        gradient_penalty.backward()

    def train_hash_network(self, h_img1, h_img2, h_txt1, h_txt2, ind, e_losses_dict, weights=1):
        """
        Train hashing network

        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)
        :param: h_txt1: batch of text hashes #1 (original)
        :param: h_txt2: batch of text hashes #2 (augmented)
        :param: ind: indexes of samples in current batch
        :param: e_losses_dict: dictionary with epoch losses
        :param: weights: weights for pairs (true pair or not)

        :returns: updated dictionary with epoch losses
        """
        err, batch_losses_dict = self.calculate_losses(h_img1, h_img2, h_txt1, h_txt2, ind, weights)
        e_losses_dict = self.update_epoch_losses_dict(e_losses_dict, batch_losses_dict)

        self.optimizer_gen.zero_grad()
        err.backward()
        self.optimizer_gen.step()

        return e_losses_dict

    def calculate_losses(self, h_img1, h_img2, h_txt1, h_txt2, ind, weights):
        """
        Calculate losses
        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)
        :param: h_txt1: batch of text hashes #1 (original)
        :param: h_txt2: batch of text hashes #2 (augmented)
        :param: ind: indexes of samples in current batch
        :param: weights: weights for pairs (true pair or not)

        :returns: total epoch loss and updated dictionary with epoch losses
        """

        loss_ntxent = self.calc_ntxent_loss(h_img1, h_img2, h_txt1, h_txt2, weights)
        loss_quant = self.calc_quantization_loss(h_img1, h_img2, h_txt1, h_txt2, ind) * self.cfg.beta

        err = loss_ntxent + loss_quant

        e_losses = self.get_batch_losses_dict(loss_ntxent, loss_quant)

        return err, e_losses

    def calc_ntxent_loss(self, h_img1, h_img2, h_txt1, h_txt2, weights):
        """
        Calculate NTXent Loss

        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)
        :param: h_txt1: batch of text hashes #1 (original)
        :param: h_txt2: batch of text hashes #2 (augmented)
        :param: weights: weights for pairs (true pair or not)

        :returns: NTXent Loss
        """
        if type(weights) is torch.Tensor:
            intra_ws = float(weights.mean().detach().cpu())
        else:
            intra_ws = 1

        loss_ntxent_inter = self.contr_loss(h_img1, h_txt1, weights, type='orig') * self.cfg.contrastive_weights[0]
        loss_ntxent_intra_img = self.contr_loss(h_img1, h_img2, intra_ws, type='orig') * self.cfg.contrastive_weights[1]
        loss_ntxent_intra_txt = self.contr_loss(h_txt1, h_txt2, intra_ws, type='orig') * self.cfg.contrastive_weights[2]
        loss_ntxent = loss_ntxent_inter + loss_ntxent_intra_txt + loss_ntxent_intra_img
        return loss_ntxent

    def calc_adversarial_loss(self, h_img1, h_img2):
        """
        Calculate Adversarial Loss

        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)

        :returns: Adversarial Loss
        """
        loss_adv_h1 = -torch.log(torch.sigmoid(self.model.discriminate_hash(h_img1))).sum()
        loss_adv_h2 = -torch.log(torch.sigmoid(self.model.discriminate_hash(h_img2))).sum()
        loss_adv_h = (loss_adv_h1 + loss_adv_h2)
        return loss_adv_h

    def calc_quantization_loss(self, h_img1, h_img2, h_txt1, h_txt2, ind):
        """
        Calculate Quantization Loss

        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)
        :param: h_txt1: batch of text hashes #1 (original)
        :param: h_txt2: batch of text hashes #2 (augmented)

        :returns: Quantization Loss
        """
        loss_quant_img1 = torch.sum(torch.pow(self.B[ind, :] - h_img1, 2))
        loss_quant_img2 = torch.sum(torch.pow(self.B[ind, :] - h_img2, 2))
        loss_quant_txt1 = torch.sum(torch.pow(self.B[ind, :] - h_txt1, 2))
        loss_quant_txt2 = torch.sum(torch.pow(self.B[ind, :] - h_txt2, 2))
        loss_quant = (loss_quant_img1 + loss_quant_img2 + loss_quant_txt1 + loss_quant_txt2)
        return loss_quant

    def calc_bit_balance_loss(self, h_img1, h_img2, h_txt1, h_txt2):
        """
        Calculate Bit Balance Loss

        :param: h_img1: batch of image hashes #1 (original)
        :param: h_img2: batch of image hashes #2 (augmented)
        :param: h_txt1: batch of text hashes #1 (original)
        :param: h_txt2: batch of text hashes #2 (augmented)

        :returns: Bit Balance Loss
        """
        # loss_bb_img1 = torch.sum(torch.pow(h_img1, 2))
        # loss_bb_img2 = torch.sum(torch.pow(h_img2, 2))
        # loss_bb_txt1 = torch.sum(torch.pow(h_txt1, 2))
        # loss_bb_txt2 = torch.sum(torch.pow(h_txt2, 2))
        loss_bb_img1 = torch.sum(torch.pow(torch.sum(h_img1, dim=1), 2))
        loss_bb_img2 = torch.sum(torch.pow(torch.sum(h_img2, dim=1), 2))
        loss_bb_txt1 = torch.sum(torch.pow(torch.sum(h_txt1, dim=1), 2))
        loss_bb_txt2 = torch.sum(torch.pow(torch.sum(h_txt2, dim=1), 2))
        loss_bb = (loss_bb_img1 + loss_bb_img2 + loss_bb_txt1 + loss_bb_txt2)
        return loss_bb

    @staticmethod
    def get_batch_losses_dict(loss_ntxent, loss_quant):
        """
        Get batch losses dict

        :param: loss_adv_h: Adversarial Loss
        :param: loss_ntxent: NTXent Loss
        :param: loss_quant: Quantization Loss
        :param: loss_bb: Bit Balance Loss

        :returns: batch losses dict
        """
        b_losses_dict = {'quant': 0., 'ntxent': 0.}
        b_losses_dict['ntxent'] += loss_ntxent.detach().cpu().numpy()
        b_losses_dict['quant'] += loss_quant.detach().cpu().numpy()
        b_losses_dict['sum'] = sum(b_losses_dict.values())
        return b_losses_dict

    @staticmethod
    def update_epoch_losses_dict(e_losses_dict, b_losses_dict):
        """
        Update epoch losses dictionary

        :param: e_losses_dict: epoch losses dictionary
        :param: b_losses_dict: batch losses dictionary

        :returns: updated dictionary with epoch losses
        """
        for k, v in b_losses_dict.items():
            e_losses_dict[k] += v
        return e_losses_dict

    def update_optimizer_params(self, epoch):
        """
        Update parameters of optimizers

        :param: epoch: current epoch
        """
        if epoch % 50 == 0:
            for params in self.optimizer_gen.param_groups:
                params['lr'] = max(params['lr'] * 0.8, 1e-6)

    def eval(self, epoch):
        """
        Evaluate model. Calculate MAPs for current epoch
        Save model and hashes if current epoch is the best

        :param: epoch: current epoch
        """
        self.model.eval()

        qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes = self.get_codes_labels_indexes()

        mapi2t, mapt2i, mapi2i, mapt2t, mapavg = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY,
                                                                  self.cfg.retrieval_map_k)

        map_k_5 = (mapi2t, mapt2i, mapi2i, mapt2t, mapavg)
        map_k_10 = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 10)
        map_k_20 = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 20)
        map_r = self.calc_maps_rad(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, [0, 1, 2, 3, 4, 5])
        p_at_k = self.calc_p_top_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        maps_eval = (map_k_5, map_k_10, map_k_20, map_r, p_at_k)

        if self.cfg.build_plots:
            hr_hists(qBX, qBY, rBX, rBY, model='UNHD')
            top_k_hists(qBX, qBY, rBX, rBY, model='UNHD')
            self.visualize_retrieval(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes, 'UNHD')
            bucket_hists(qBX, qBY, rBX, rBY, 'UNHD', [i[0] for i in map_r], self.cfg.hash_dim)

        self.update_maps_dict(mapi2t, mapt2i, mapi2i, mapt2t, mapavg)

        if mapavg > self.maps_max['avg']:
            self.update_max_maps_dict(mapi2t, mapt2i, mapi2i, mapt2t, mapavg)

            self.save_model()
            self.save_hash_codes()

        self.save_model('last')
        write_pickle(os.path.join(self.path, 'maps_eval.pkl'), maps_eval)

        self.model.train()

    def visualize_retrieval(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes, file_tag):

        qIX, qIY, rIX, rIY = indexes
        cfg_params = [self.cfg.seed, self.cfg.dataset_json_file, self.cfg.dataset_image_folder_path]

        i2t = retrieval2png(cfg_params, qBX, rBY, qLX, rLY, qIX, rIY, tag='I2T', file_tag=file_tag)
        t2i = retrieval2png(cfg_params, qBY, rBX, qLY, rLX, qIY, rIX, tag='T2I', file_tag=file_tag)
        i2i = retrieval2png(cfg_params, qBX, rBX, qLX, rLX, qIX, rIX, tag='I2I', file_tag=file_tag)
        t2t = retrieval2png(cfg_params, qBY, rBY, qLY, rLY, qIY, rIY, tag='T2T', file_tag=file_tag)

    def calc_maps_k(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, k):
        """
        Calculate MAPs, in regards to K

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y
        :param: k: k

        :returns: MAPs
        """
        mapi2t = calc_map_k(qBX, rBY, qLX, rLY, k)
        mapt2i = calc_map_k(qBY, rBX, qLY, rLX, k)
        mapi2i = calc_map_k(qBX, rBX, qLX, rLX, k)
        mapt2t = calc_map_k(qBY, rBY, qLY, rLY, k)

        avg = (mapi2t.item() + mapt2i.item() + mapi2i.item() + mapt2t.item()) * 0.25

        mapi2t, mapt2i, mapi2i, mapt2t, mapavg = mapi2t.item(), mapt2i.item(), mapi2i.item(), mapt2t.item(), avg

        s = 'Valid: mAP@{:2d}, avg: {:3.3f}, i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        self.logger.info(s.format(k, mapavg, mapi2t, mapt2i, mapi2i, mapt2t))

        return mapi2t, mapt2i, mapi2i, mapt2t, mapavg

    def calc_maps_rad(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, rs):
        """
        Calculate MAPs, in regard to Hamming radius

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y
        :param: rs: hamming radiuses to output

        :returns: MAPs
        """
        mapsi2t = calc_map_rad(qBX, rBY, qLX, rLY)
        mapst2i = calc_map_rad(qBY, rBX, qLY, rLX)
        mapsi2i = calc_map_rad(qBX, rBX, qLX, rLX)
        mapst2t = calc_map_rad(qBY, rBY, qLY, rLY)

        mapsi2t, mapst2i, mapsi2i, mapst2t = mapsi2t.numpy(), mapst2i.numpy(), mapsi2i.numpy(), mapst2t.numpy()

        s = 'Valid: mAP HR{}, i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        for r in rs:
            self.logger.info(s.format(r, mapsi2t[r], mapst2i[r], mapsi2i[r], mapst2t[r]))

        return mapsi2t, mapst2i, mapsi2i, mapst2t

    def get_codes_labels_indexes(self, remove_replications=True):
        """
        Generate binary codes from duplet dataloaders for query and response

        :param: remove_replications: remove replications from dataset

        :returns: hash codes and labels for query and response, sample indexes
        """
        # hash X, hash Y, labels X/Y, image replication factor, indexes X, indexes Y
        qBX, qBY, qLXY, irf_q, (qIX, qIY) = self.generate_codes_from_dataloader(self.dataloader_q)
        # hash X, hash Y, labels X/Y, image replication factor, indexes X, indexes Y
        rBX, rBY, rLXY, irf_db, (rIX, rIY) = self.generate_codes_from_dataloader(self.dataloader_db)

        # get Y Labels
        qLY = qLXY
        rLY = rLXY

        # X modality sometimes contains replicated samples (see datasets), remove them by selecting each nth element
        # remove replications for hash codes
        qBX = self.get_each_nth_element(qBX, irf_q)
        rBX = self.get_each_nth_element(rBX, irf_db)
        # remove replications for labels
        qLX = self.get_each_nth_element(qLXY, irf_q)
        rLX = self.get_each_nth_element(rLXY, irf_db)
        # remove replications for indexes
        qIX = self.get_each_nth_element(qIX, irf_q)
        rIX = self.get_each_nth_element(rIX, irf_db)

        return qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, (qIX, qIY, rIX, rIY)

    @staticmethod
    def get_each_nth_element(arr, n):
        """
        intentionally ugly solution, needed to avoid query replications during test/validation

        :return: array
        """
        return arr[::n]

    def generate_codes_from_dataloader(self, dataloader):
        """
        Generate binary codes from duplet dataloader

        :param: dataloader: duplet dataloader

        :returns: hash codes for given duplet dataloader, image replication factor of dataset
        """
        num = len(dataloader.dataset)

        irf = dataloader.dataset.image_replication_factor

        Bi = torch.zeros(num, self.cfg.hash_dim).to(self.device)
        Bt = torch.zeros(num, self.cfg.hash_dim).to(self.device)
        L = torch.zeros(num, self.cfg.label_dim).to(self.device)

        dataloader_idxs = []

        # for i, input_data in tqdm(enumerate(test_dataloader)):
        for i, (idx, sample_idxs, img, txt, label) in enumerate(dataloader):
            dataloader_idxs = self.stack_idxs(dataloader_idxs, sample_idxs)
            img = img.to(self.device)
            txt = txt.to(self.device)
            if len(label.shape) == 1:
                label = one_hot(label, num_classes=self.cfg.label_dim).to(self.device)
            else:
                label.to(self.device)
            bi = self.model.generate_img_code(img)
            bt = self.model.generate_txt_code(txt)
            idx_end = min(num, (i + 1) * self.cfg.batch_size)
            Bi[i * self.cfg.batch_size: idx_end, :] = bi.data
            Bt[i * self.cfg.batch_size: idx_end, :] = bt.data
            L[i * self.cfg.batch_size: idx_end, :] = label.data

        Bi = torch.sign(Bi)
        Bt = torch.sign(Bt)
        return Bi, Bt, L, irf, dataloader_idxs

    @staticmethod
    def stack_idxs(idxs, idxs_batch):
        if len(idxs) == 0:
            return [ib for ib in idxs_batch]
        else:
            return [torch.hstack(i).detach() for i in zip(idxs, idxs_batch)]

    def update_maps_dict(self, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
        """
        Update MAPs dictionary (append new values)

        :param: mapi2t: I-to-T MAP
        :param: mapt2i: T-to-I MAP
        :param: mapi2i: I-to-I MAP
        :param: mapt2t: T-to-T MAP
        :param: mapavg: average MAP
        """
        self.maps['i2t'].append(mapi2t)
        self.maps['t2i'].append(mapt2i)
        self.maps['i2i'].append(mapi2i)
        self.maps['t2t'].append(mapt2t)
        self.maps['avg'].append(mapavg)

    def update_max_maps_dict(self, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
        """
        Update max MAPs dictionary (replace values)

        :param: mapi2t: I-to-T MAP
        :param: mapt2i: T-to-I MAP
        :param: mapi2i: I-to-I MAP
        :param: mapt2t: T-to-T MAP
        :param: mapavg: average MAP
        """
        self.maps_max['i2t'] = mapi2t
        self.maps_max['t2i'] = mapt2i
        self.maps_max['i2i'] = mapi2i
        self.maps_max['t2t'] = mapt2t
        self.maps_max['avg'] = mapavg

    def save_hash_codes(self):
        """
        Save hash codes on a disk
        """
        with torch.cuda.device(self.device):
            torch.save([self.Hi1, self.Hi2, self.Ht1, self.Ht2], os.path.join(self.path, 'hash_codes_i_t.pth'))
        with torch.cuda.device(self.device):
            torch.save(self.B, os.path.join(self.path, 'hash_code.pth'))

    def training_complete(self):
        """
        Output training summary: time and best results
        """
        self.save_train_results_dict()

        current = time.time()
        delta = current - self.since
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(delta // 60, delta % 60))
        s = 'Max Avg MAP: {avg:3.3f}, Max MAPs: i->t: {i2t:3.3f}, t->i: {t2i:3.3f}, i->i: {i2i:3.3f}, t->t: {t2t:3.3f}'
        self.logger.info(s.format(**self.maps_max))

    def save_train_results_dict(self):
        """
        Save training history dictionary
        """
        res = self.maps
        res['losses'] = self.losses
        res['probabs'] = self.probabs
        write_pickle(os.path.join(self.path, 'train_res_dict.pkl'), res)

    def test(self):
        """
        Test model. Calculate MAPs, PR-curves and P@K values.
        """
        self.model.to(self.device).train()

        self.model.eval()

        qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes = self.get_codes_labels_indexes()

        mapi2t, mapt2i, mapi2i, mapt2t, mapavg = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 5)

        #map_k_5 = (mapi2t, mapt2i, mapi2i, mapt2t, mapavg)
        #map_k_10 = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 10)
        #map_k_20 = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 20)
        #map_r = self.calc_maps_rad(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, [0, 1, 2, 3, 4, 5])
        #p_at_k = self.calc_p_top_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        #maps_eval = (map_k_5, map_k_10, map_k_20, map_r, p_at_k)

        #hr_hists(qBX, qBY, rBX, rBY, model='UNHD')
        #top_k_hists(qBX, qBY, rBX, rBY, model='UNHD')
        self.visualize_retrieval(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes, 'UNHD')
        #bucket_hists(qBX, qBY, rBX, rBY, 'UNHD', [i[0] for i in map_r], self.cfg.hash_dim)

        """
        maps = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, self.cfg.retrieval_map_k)
        map_dict = self.make_maps_dict(*maps)
        pr_dict = self.calc_pr_curves(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        pk_dict = self.calc_p_top_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        self.save_test_results_dicts(map_dict, pr_dict, pk_dict)
        """

        self.model.train()

        current = time.time()
        delta = current - self.since
        self.logger.info('Test complete in {:.0f}m {:.0f}s'.format(delta // 60, delta % 60))

    def make_maps_dict(self, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
        """
        Make MAP dict from MAP values

        :param: mapi2t: I-to-T MAP
        :param: mapt2i: T-to-I MAP
        :param: mapi2i: I-to-I MAP
        :param: mapt2t: T-to-T MAP
        :param: mapavg: Average MAP

        :returns: MAPs dictionary
        """

        map_dict = {'mapi2t': mapi2t, 'mapt2i': mapt2i, 'mapi2i': mapi2i, 'mapt2t': mapt2t, 'mapavg': mapavg}

        s = 'Avg MAP: {:3.3f}, MAPs: i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        self.logger.info(s.format(mapavg, mapi2t, mapt2i, mapi2i, mapt2t))

        return map_dict

    def calc_pr_curves(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY):
        """
        Calculate PR-curves

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y

        :returns: PR-curves dictionary
        """
        p_i2t, r_i2t = pr_curve(qBX, rBY, qLX, rLY, tqdm_label='I2T')
        p_t2i, r_t2i = pr_curve(qBY, rBX, qLY, rLX, tqdm_label='T2I')
        p_i2i, r_i2i = pr_curve(qBX, rBX, qLX, rLX, tqdm_label='I2I')
        p_t2t, r_t2t = pr_curve(qBY, rBY, qLY, rLY, tqdm_label='T2T')

        pr_dict = {'pi2t': p_i2t, 'ri2t': r_i2t,
                   'pt2i': p_t2i, 'rt2i': r_t2i,
                   'pi2i': p_i2i, 'ri2i': r_i2i,
                   'pt2t': p_t2t, 'rt2t': r_t2t}

        self.logger.info('Precision-recall values: {}'.format(pr_dict))

        return pr_dict

    def calc_p_top_k(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY):
        """
        Calculate P@K values

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y

        :returns: P@K values
        """
        k = [1, 5, 10, 20, 50] + list(range(100, 1001, 100))

        pk_i2t = p_top_k(qBX, rBY, qLX, rLY, k, tqdm_label='I2T')
        pk_t2i = p_top_k(qBY, rBX, qLY, rLX, k, tqdm_label='T2I')
        pk_i2i = p_top_k(qBX, rBX, qLX, rLX, k, tqdm_label='I2I')
        pk_t2t = p_top_k(qBY, rBY, qLY, rLY, k, tqdm_label='T2T')

        pk_dict = {'k': k,
                   'pki2t': pk_i2t,
                   'pkt2i': pk_t2i,
                   'pki2i': pk_i2i,
                   'pkt2t': pk_t2t}

        self.logger.info('P@K values: {}'.format(pk_dict))

        return pk_dict

    def save_test_results_dicts(self, map_dict, pr_dict, pk_dict):
        """
        Save test results dictionary

        :param: map_dict: MAPs dictionary
        :param: pr_dict: PR-curves dictionary
        :param: pk_dict: P@K values dictionary
        """
        write_pickle(os.path.join(self.path, 'map_dict.pkl'), map_dict)
        write_pickle(os.path.join(self.path, 'pr_dict.pkl'), pr_dict)
        write_pickle(os.path.join(self.path, 'pk_dict.pkl'), pk_dict)

    def load_model(self, tag='best'):
        """
        Load model from the disk

        :param: tag: name tag
        """
        self.model.load(os.path.join(self.path, self.model.module_name + '_' + str(tag) + '.pth'))

    def save_model(self, tag='best'):
        """
        Save model on the disk

        :param: tag: name tag
        """
        self.model.save(self.model.module_name + '_' + str(tag) + '.pth', self.path, cuda_device=self.device)
