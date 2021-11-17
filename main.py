from utils import logger
from controller_UNHD import ControllerUNHD
from dataset.data_handler import get_dataloaders, DataHandler, DataHandlerAugmentedTxt, DataHandlerAugmentedTxtImg
from dataset.dataset_ucm import *
from configs.config import cfg


def main():
    log = logger()

    mode = 'TEST' if cfg.test else 'TRAIN'
    s = 'Init ({}): {}, {}, {} bits, tag: {}, preset: {}'
    log.info(s.format(mode, cfg.model_type, cfg.dataset.upper(), cfg.hash_dim, cfg.tag, cfg.preset))

    # Training with img-img_aug-txt-txt_aug
    data_handler_class = DataHandlerAugmentedTxtImg
    ds_train_class = DatasetQuadrupletAugmentedTxtImgNoiseWrongCaption
    ds_train_class_clean = DatasetQuadrupletAugmentedTxtImgNoiseWrongCaptionClean
    ds_query_class = DatasetDuplet1
    ds_db_class = DatasetDuplet1
    controller_class = ControllerUNHD

    dl_train, dl_train_clean, dl_q, dl_db = get_dataloaders(data_handler_class, ds_train_class, ds_train_class_clean, ds_query_class, ds_db_class)

    controller = controller_class(log, cfg, (dl_train, dl_train_clean, dl_q, dl_db))

    if cfg.test:
        controller.load_model()
        controller.test()
    else:
        for epoch in range(cfg.max_epoch):
            if epoch < cfg.clean_epochs:
                controller.train_epoch_clean_data(epoch)
            else:
                controller.train_epoch(epoch)
            if (((epoch + 1) % cfg.valid_freq == 0) or ((epoch + 1) == cfg.clean_epochs)) and cfg.valid:
                controller.eval(epoch)
            # save the model
            if epoch + 1 == cfg.max_epoch:
                controller.save_model('last')
                controller.training_complete()


if __name__ == '__main__':
    main()
