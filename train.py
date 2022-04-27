from models.mesh_quantil_classifier import CFDQuantileAccuracy
from options.test_options import TestOptions
from options.train_options import TrainOptions
from models.mesh_regression import CFDLoss
from data import DataLoader
from models import create_model
import os
import numpy as np
import wandb
import torch
from util.util import clear_mesh_cashes
from util.wandb_logger import WandbLogger


def train_regression_model(model, train_data_loader, test_data_loader, logger, train_opt, test_opt):
    for epoch in range(300):
        print(f"Epoch {epoch}:")

        # train
        training_loss = CFDLoss(0, 0, 0)
        train_batches = 0
        for i, data in enumerate(train_data_loader):
            model.set_input(data)
            model.optimize_parameters()
            training_loss += model.loss_value
            train_batches += 1

        # log training loss
        training_loss /= train_batches
        logger.log_train_av_loss(training_loss, epoch)
        model.save_network(epoch)
        logger.log_model(epoch)
        print(f"Train Loss: {str(training_loss)}")

        # test
        if epoch % train_opt.run_test_freq == 0:

            test_opt.which_epoch = epoch
            test_model = create_model(test_opt)
            logger.init_log_prediction()
            test_loss = CFDLoss(0, 0, 0)
            test_batches = 0
            for i, data in enumerate(test_data_loader):
                test_model.set_input(data)
                test_model.test()
                test_loss += test_model.loss_value
                test_batches += 1

                if test_opt.verbose_test:
                    samples = [("cda", test_model.cda_out.cpu().numpy(), test_model.labels_cda.unsqueeze(1).cpu().numpy()),
                               ("cla", test_model.cla_out.cpu().numpy(), test_model.labels_cla.unsqueeze(1).cpu().numpy()),
                               ("cop", test_model.cop_out.cpu().numpy(), test_model.labels_cop.cpu().numpy())]
                    for (name, pred, label) in samples:
                        if train_opt.normalise_targets:
                            pred_unormalised = test_data_loader.dataset.un_normalise_target(pred[0], name)
                            label_unormalised = test_data_loader.dataset.un_normalise_target(label[0], name)
                        else:
                            pred_unormalised = pred[0]
                            label_unormalised = label[0]

                        logger.log_predictions(epoch, test_batches, name, pred_unormalised,label_unormalised)

            # log test loss
            test_loss /= test_batches
            print(f"Test Loss: {str(test_loss)}")
            logger.commit_predictions()
            logger.log_test_av_loss(test_loss, epoch)

        model.update_learning_rate()
        print("-----------------------------------------------\n")


def run_experiment():
    train_opt = TrainOptions().parse()
    logger = WandbLogger(train_opt)
    logger.initialise_logging()

    # parse train options and create train dataset
    train_dataset = DataLoader(train_opt)

    # parse test options and create test dataset
    test_opt = TestOptions().parse()
    test_opt.serial_batches = True  # no shuffle
    test_dataset = DataLoader(test_opt)

    # clear caches
    clear_mesh_cashes(train_opt.dataroot)
    torch.cuda.empty_cache()
    print("Cleared cuda cache")
    # create model
    model = create_model(train_opt)

    train_regression_model(model, train_dataset, test_dataset, logger, train_opt, test_opt)



if __name__ == '__main__':
    run_experiment()
