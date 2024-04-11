"""
This script defines a Flower Client for federated learning using CycleGAN. The FlwrClient class inherits from
fl.client.NumPyClient and implements the required methods for federated learning, including get_parameters,
set_parameters, fit, and evaluate.

The FlwrClient class is initialized with a client id, options, and a dataloader. It sets up the CycleGAN model and a
visualizer for displaying and saving images and plots.

The get_parameters method retrieves the parameters of the CycleGAN model's generators and discriminators. The
set_parameters method sets the parameters of the CycleGAN model's generators and discriminators.

The fit method trains the CycleGAN model for a specified number of epochs, updating the model's parameters,
calculating loss functions, getting gradients, and updating network weights. It also saves the model's state at
specified intervals.

The evaluate method is a placeholder and currently returns a fixed accuracy and loss.

The generate_client_fn function returns a function that can be used by the VirtualClientEngine to spawn a FlwrClient
with a specified client id."""
import time
from models import create_model
from util.visualizer import Visualizer

import flwr as fl
from collections import OrderedDict
from typing import List
import numpy as np
import torch


class FlwrClient(fl.client.NumPyClient):

    def __init__(self, cid, opt, dataloader):
        self.opt = opt
        self.opt.cid = cid
        self.dataloader = dataloader
        self.dataset_size = len(self.dataloader)  # get the number of images in the dataset.
        print('The number of training images = %d' % self.dataset_size)

        self.model = create_model(opt)  # create a model given opt.model and other options
        self.model.setup(opt)
        self.visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.netG_A.state_dict().items()] + \
            [val.cpu().numpy() for _, val in self.model.netG_B.state_dict().items()] + \
            [val.cpu().numpy() for _, val in self.model.netD_A.state_dict().items()] + \
            [val.cpu().numpy() for _, val in self.model.netD_B.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        len_g_a_param = len([val.cpu().numpy() for _, val in self.model.netG_A.state_dict().items()])
        len_g_b_param = len([val.cpu().numpy() for _, val in self.model.netG_B.state_dict().items()])
        len_d_a_param = len([val.cpu().numpy() for _, val in self.model.netD_A.state_dict().items()])
        params_dict = zip(self.model.netG_A.state_dict().keys(), parameters[:len_g_a_param])
        net_g_a_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.netG_A.load_state_dict(net_g_a_state_dict, strict=True)

        params_dict = zip(self.model.netG_B.state_dict().keys(),
                          parameters[len_g_a_param:len_g_a_param + len_g_b_param])
        net_g_b_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.netG_B.load_state_dict(net_g_b_state_dict, strict=False)

        params_dict = zip(self.model.netD_A.state_dict().keys(), parameters[
                                                                 len_g_a_param + len_g_b_param:
                                                                 len_g_a_param + len_g_b_param + len_d_a_param])
        net_d_a_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.netD_A.load_state_dict(net_d_a_state_dict, strict=False)

        params_dict = zip(self.model.netD_B.state_dict().keys(), parameters[
                                                                 len_g_a_param + len_g_b_param + len_d_a_param:])
        net_d_b_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.netD_B.load_state_dict(net_d_b_state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.save_networks("FL-Round")
        total_iters = 0  # the total number of training iterations
        for epoch in range(self.opt.epoch_count,
                           self.opt.n_epochs + self.opt.n_epochs_decay + 1):  # outer loop for different epochs;
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            self.visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once
            # every epoch
            self.model.update_learning_rate()  # update learning rates in the beginning of every epoch.
            for data in self.dataloader:  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % self.opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += self.opt.batch_size
                epoch_iter += self.opt.batch_size
                self.model.set_input(data)  # unpack data from dataset and apply preprocessing
                self.model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                if total_iters % self.opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = total_iters % self.opt.update_html_freq == 0
                    self.model.compute_visuals()
                    self.visualizer.display_current_results(self.model.get_current_visuals(), epoch, save_result)

                if total_iters % self.opt.print_freq == 0:  # print training losses and save logging information to
                    # the disk
                    losses = self.model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / self.opt.batch_size
                    self.visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if self.opt.display_id > 0:
                        self.visualizer.plot_current_losses(epoch, float(epoch_iter) / self.dataset_size, losses)

                if total_iters % self.opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq>
                    # iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if self.opt.save_by_iter else 'latest'
                    self.model.save_networks(save_suffix)

                iter_data_time = time.time()
            if epoch % self.opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                self.model.save_networks('latest')
                self.model.save_networks(epoch)

            self.visualizer.print_epoch_log(epoch, time.time() - epoch_start_time)
        return self.get_parameters(config={}), self.dataset_size, {}

    def evaluate(self, parameters, config):
        # self.set_parameters(parameters)
        return float(1), 1, {}


def generate_client_fn(opt, dataloader):
    """Return a function that can be used by the VirtualClientEngine.
    ti spawn a FlwrClient with client id 'cid'"""

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())
        return FlwrClient(
            cid,
            opt,
            dataloader[int(cid)]
        ).to_client()

    return client_fn
