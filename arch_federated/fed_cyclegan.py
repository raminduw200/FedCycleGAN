import time
from models import create_model
from data import create_dataset
from arch_federated.server import aggregate_from_client_to_server, update_server_from_best_psnr_client
from arch_federated.client import send_from_server_to_client
from util.visualizer import Visualizer

__all__ = ['FedCycleGAN']


class FedCycleGAN:
    def __init__(self, opt):
        self.opt = opt
        # data

        # server and client model
        self.server = None
        self.clients = []
        self.client_num = opt.client_num

        # global model
        self.server_gener = {'from_a_to_b': None, 'from_b_to_a': None}
        self.server_discr = {'from_a_to_b': None, 'from_b_to_a': None}
        self.client_gener_list = {'from_a_to_b': [], 'from_b_to_a': []}
        self.client_discr_list = {'from_a_to_b': [], 'from_b_to_a': []}
        self.client_psnr_list = []

        # save and log
        # save model
        self.best_psnr = 0

        self.visualizer = Visualizer(self.opt)  # create a visualizer that display/save images and plots

    def init_model(self, description='server and clients'):
        self.clients = []

        for i in range(self.client_num):
            client = create_model(self.opt, id=i, is_client=True)  # create a model given opt.model and other options
            client.setup(self.opt)
            self.clients.append(client)
        self.server = create_model(self.opt, id=0, is_client=False)  # create a model given opt.model and other options
        self.server.setup(self.opt)

    def clients_training(self):
        for i in range(self.client_num):
            print(f"\n\ntraining client: {i + 1}")

            total_iters = 0  # the total number of training iterations
            if 'client_' in self.opt.dataroot.split('/')[-1]:
                self.opt.dataroot = "/".join(self.opt.dataroot.split('/')[:-1])
            self.opt.dataroot += f'/client_{i+1}'
            dataset = create_dataset(self.opt)  # create a dataset given opt.dataset_mode and other options
            dataset_size = len(dataset)  # get the number of images in the dataset.
            print('The number of training images = %d' % dataset_size)
            self.visualizer.reset(i, False)  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

            for epoch in range(self.opt.epoch_count,
                               self.opt.n_epochs + self.opt.n_epochs_decay + 1):  # outer loop for different epochs;
                # we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
                epoch_start_time = time.time()  # timer for entire epoch
                iter_data_time = time.time()  # timer for data loading per iteration
                epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
                self.visualizer.reset(i, False)  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
                self.clients[i].update_learning_rate(
                    self.opt.lr_policy)  # update learning rates in the beginning of every epoch.
                for _, data in enumerate(dataset):  # inner loop within one epoch
                    iter_start_time = time.time()  # timer for computation per iteration
                    if total_iters % self.opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time

                    total_iters += self.opt.batch_size
                    epoch_iter += self.opt.batch_size
                    self.clients[i].set_input(data)  # unpack data from dataset and apply preprocessing
                    self.clients[i].optimize_parameters()  # calculate loss functions, get gradients, update network
                    # weights

                    if total_iters % self.opt.display_freq == 0:  # display images on visdom and save images to a
                        # HTML file
                        save_result = total_iters % self.opt.update_html_freq == 0
                        self.clients[i].compute_visuals()
                        self.visualizer.display_current_results(self.clients[i].get_current_visuals(), epoch,
                                                                save_result)

                    if total_iters % self.opt.print_freq == 0:  # print training losses and save logging information to
                        # the disk
                        losses = self.clients[i].get_current_losses()
                        # metrics = self.clients[i].get_eval_metrics()  # get metrics
                        t_comp = (time.time() - iter_start_time) / self.opt.batch_size
                        self.visualizer.print_current_losses(i, epoch, epoch_iter, losses, t_comp, t_data)
                        if self.opt.display_id > 0:
                            self.visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                #     if total_iters % self.opt.save_latest_freq == 0:  # cache our latest model every
                #         # <save_latest_freq> iterations
                #         print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                #         save_suffix = 'iter_%d' % total_iters if self.opt.save_by_iter else 'latest'
                #         self.clients[i].save_networks(save_suffix)
                #
                #     iter_data_time = time.time()
                # if epoch % self.opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                #     self.clients[i].save_networks('latest')
                #     self.clients[i].save_networks(epoch)

                print('End of epoch %d / %d \t Time Taken: %d sec\n' % (
                    epoch, self.opt.n_epochs + self.opt.n_epochs_decay, time.time() - epoch_start_time))

    # def save_models(self, psnr):
    #     if self.para_dict['model'] == 'cyclegan':
    #         save_model(self.server_gener['from_a_to_b'], '{}/checkpoint/g_from_a_to_b'.format(self.file_path),
    #                    self.para_dict, psnr)
    #         save_model(self.server_gener['from_b_to_a'], '{}/checkpoint/g_from_b_to_a'.format(self.file_path),
    #                    self.para_dict, psnr)
    #         save_model(self.server_discr['from_a_to_b'], '{}/checkpoint/d_from_a_to_b'.format(self.file_path),
    #                    self.para_dict, psnr)
    #         save_model(self.server_discr['from_b_to_a'], '{}/checkpoint/d_from_b_to_a'.format(self.file_path),
    #                    self.para_dict, psnr)

    # def server_inference(self):
    #     # evaluation from a to b
    #     mae, psnr, ssim, fid = self.server.evaluation(direction='from_a_to_b')
    #     infor = '[Round {}/{}] [{} -> {}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f}'.format(
    #         self.round + 1, self.para_dict['num_round'], self.para_dict['source_domain'],
    #         self.para_dict['target_domain'], mae, psnr, ssim)
    #     print(infor)
    #
    #     if self.para_dict['save_log']:
    #         save_log(infor, self.file_path, description='_server_from_a_to_b')
    #
    #     # evaluation from b to a
    #     mae, psnr, ssim, fid = self.server.evaluation(direction='from_b_to_a')
    #     infor = '[Round {}/{}] [{} -> {}] mae: {:.4f} psnr: {:.4f} ssim: {:.4f}'.format(
    #         self.round + 1, self.para_dict['num_round'], self.para_dict['target_domain'],
    #         self.para_dict['source_domain'], mae, psnr, ssim)
    #     print(infor)
    #
    #     if self.para_dict['save_log']:
    #         save_log(infor, self.file_path, description='_server_from_b_to_a')
    #
    #     if self.para_dict['plot_distribution']:
    #         save_img_path = '{}/sample_distribution'.format(self.file_path)
    #         if not os.path.exists(save_img_path):
    #             os.makedirs(save_img_path)
    #         save_img_path = '{}/round_{}.png'.format(save_img_path, self.round + 1)
    #         self.server.visualize_feature(self.round + 1, save_img_path, self.train_loader)
    #
    #     if self.para_dict['save_img']:
    #         save_img_path = '{}/images/round_{}'.format(self.file_path, self.round + 1)
    #         if not os.path.exists(save_img_path):
    #             os.makedirs(save_img_path)
    #         self.server.infer_images(save_img_path, self.valid_loader)
    #
    #     if self.para_dict['save_model']:
    #         if psnr > self.best_psnr:
    #             self.save_models(psnr)
    #             self.best_psnr = psnr

    def collect_model(self, description='server and clients'):
        self.server_gener['from_a_to_b'] = self.server.netG_A
        self.server_gener['from_b_to_a'] = self.server.netG_B

        self.server_discr['from_a_to_b'] = self.server.netD_A
        self.server_discr['from_b_to_a'] = self.server.netD_B

        # clear old weights
        self.client_gener_list = {'from_a_to_b': [], 'from_b_to_a': []}
        for i in range(self.client_num):
            self.client_gener_list['from_a_to_b'].append(self.clients[i].netG_A)
            self.client_gener_list['from_b_to_a'].append(self.clients[i].netG_B)

            self.client_discr_list['from_a_to_b'].append(self.clients[i].netD_A)
            self.client_discr_list['from_b_to_a'].append(self.clients[i].netD_B)

    def aggregate_model(self, round):
        # if self.para_dict['fed_aggregate_method'] == 'fed-avg':
        aggregate_from_client_to_server(self.server_gener['from_a_to_b'], self.client_gener_list['from_a_to_b'],
                                        aggregation_weights=[0.5, 0.5])
        aggregate_from_client_to_server(self.server_gener['from_b_to_a'], self.client_gener_list['from_b_to_a'],
                                        aggregation_weights=[0.5, 0.5])
        print(f'saving the {round} central model')
        self.server.save_networks(round)

        # elif self.para_dict['fed_aggregate_method'] == 'fed-psnr':
        #     update_server_from_best_psnr_client(
        #         self.server_gener['from_a_to_b'], self.client_gener_list['from_a_to_b'], self.client_psnr_list)
        #     update_server_from_best_psnr_client(
        #         self.server_gener['from_b_to_a'], self.client_gener_list['from_b_to_a'], self.client_psnr_list)
        #     # clear
        #     self.client_psnr_list = []
        # else:
        #     raise ValueError

    def transmit_model(self):
        send_from_server_to_client(
            self.server_gener['from_a_to_b'], self.client_gener_list['from_a_to_b'])
        send_from_server_to_client(
            self.server_gener['from_b_to_a'], self.client_gener_list['from_b_to_a'])
        # for client in self.clients:

    def run_work_flow(self):
        self.init_model()

        if self.opt.continue_train:
            print('transmitting central model to clients...')
            self.collect_model()
            self.transmit_model()

        for round in range(self.opt.num_fed_rounds):
            print(f"\n---------------------- Federated Round {round + 1} ---------------------- \n")
            # train clients model
            self.clients_training()
            # collect models of clients when client model training is finished
            self.collect_model()
            # update server model
            self.aggregate_model(round)
            # test server performance and infer images for viualization
            # self.server_inference()
            # update client model
            self.transmit_model()
            print(f"\n---------------------- END Federated Round {round + 1} ---------------------- ")

        # print('work dir {}'.format(self.file_path))
        # with open('{}/log_finished.txt'.format(self.para_dict['work_dir']), 'a') as f:
        #     print('\n---> work dir {}'.format(self.file_path), file=f)
        #     print(self.args, file=f)
