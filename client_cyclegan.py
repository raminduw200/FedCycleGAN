"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time

import flwr as fl

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from typing import List
import numpy as np


class ClientCycleGAN(fl.client.NumPyClient):
    def __init__(self, opt):
        self.opt = opt
        self.dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

        self.model = create_model(opt, 99)  # create a model given opt.model and other options
        self.model.setup(opt)  # regular setup: load and print networks; create schedulers
        # self.visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots

    def get_parameters(self, config) -> List[np.ndarray]:
        return self.model.get_parameters(config)

    def set_parameters(self, parameters):
        # parameters_dict = {
        #     'G_A': parameters[0],
        #     'G_B': parameters[1],
        #     'D_A': parameters[2],
        #     'D_B': parameters[3]
        # }
        # self.model.set_parameters(parameters_dict)
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        self.model.set_parameters(parameters)
        dataset_size = len(self.dataset)  # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)
        # total_iters = 0  # the total number of training iterations
        for epoch in range(self.opt.epoch_count, self.opt.n_epochs + self.opt.n_epochs_decay + 1):  # outer loop
            # for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            # epoch_start_time = time.time()  # timer for entire epoch
            # iter_data_time = time.time()  # timer for data loading per iteration
            # epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            # self.visualizer.reset(0)  # reset the visualizer: make sure it saves the results to HTML at
            # least once every epoch
            self.model.update_learning_rate(self.opt.lr_policy)  # update learning rates in the beginning of every
            # epoch.
            for i, data in enumerate(self.dataset):  # inner loop within one epoch
                # iter_start_time = time.time()  # timer for computation per iteration
                # if total_iters % self.opt.print_freq == 0:
                #     t_data = iter_start_time - iter_data_time

                # total_iters += self.opt.batch_size
                # epoch_iter += self.opt.batch_size
                self.model.set_input(data)  # unpack data from dataset and apply preprocessing
                self.model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                # if total_iters % self.opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                #     save_result = total_iters % self.opt.update_html_freq == 0
                #     self.model.compute_visuals()
                #     self.visualizer.display_current_results(self.model.get_current_visuals(), epoch, save_result)

                # if total_iters % self.opt.print_freq == 0:  # print training losses and save logging information to
                #     # the disk
                #     losses = self.model.get_current_losses()
                #     t_comp = (time.time() - iter_start_time) / self.opt.batch_size
                #     metrics = self.model.get_eval_metrics()  # get metrics
                #     self.visualizer.print_current_losses(-1, epoch, epoch_iter, losses, t_comp, t_data, metrics)
                #     if self.opt.display_id > 0:
                #         self.visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                # if total_iters % self.opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq>
                #     # iterations
                #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                #     save_suffix = 'iter_%d' % total_iters if self.opt.save_by_iter else 'latest'
                #     self.model.save_networks(save_suffix)

                # iter_data_time = time.time()
            # if epoch % self.opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            #     self.model.save_networks('latest')
            #     self.model.save_networks(epoch)

            # print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, self.opt.n_epochs + self.opt.n_epochs_decay,
            #                                                       time.time() - epoch_start_time))
        return self.model.get_parameters(config={}), {}

    def evaluate(self, parameters, config):
        # self.model.set_parameters(parameters)
        # return self.model.get_eval_metrics()
        return float(1), {}


# Start the client
opt = TrainOptions().parse()  # get training options
fl.client.start_client(server_address="localhost:8080", client=ClientCycleGAN(opt).to_client())
