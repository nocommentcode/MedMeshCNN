import argparse
import os
from util import util
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def get_device(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        return [0]

    def initialize(self):
        # data params
        self.parser.add_argument('--dataroot', help='path to meshes (should have subfolders train, test)')
        self.parser.add_argument('--dataset_mode', choices={"classification", "segmentation", "regression", "quantile_classification"}, default='classification')
        self.parser.add_argument('--ninput_edges', type=int, default=750, help='# of input edges (will include dummy edges)')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples per epoch')
        self.parser.add_argument('--split_frac', type=float, default=0.7, help='desired split for train/test data (e.g 0.7 = 70/30 train/test)')
        self.parser.add_argument('--rebuild_meshes', action='store_true', help='if true, will rebuild meshes, and not use cache')

        # network params
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument('--arch', type=str, default='mconvnet', help='selects network to use') #todo add choices
        self.parser.add_argument('--resblocks', type=int, default=0, help='# of res blocks')
        self.parser.add_argument('--fc_n', type=int, default=100, help='# between fc and nclasses') #todo make generic
        self.parser.add_argument('--do_c', type=float, default=None, help='droput rate after convolution') #todo make generic
        self.parser.add_argument('--do_p', type=float, default=None, help='droput rate after pooling') #todo make generic
        self.parser.add_argument('--do_fc', type=float, default=None, help='droput rate after fc') #todo make generic
        self.parser.add_argument('--ncf', nargs='+', default=[16, 32, 32], type=int, help='conv filters')
        self.parser.add_argument('--nclasses', nargs='+', default=[2, 2, 2], type=int, help='number of quantile classes for cda, cla and cop')
        self.parser.add_argument('--pool_res', nargs='+', default=[1140, 780, 580], type=int, help='pooling res')
        self.parser.add_argument('--norm', type=str, default='batch',help='instance normalization or batch normalization or group normalization')
        self.parser.add_argument('--num_groups', type=int, default=16, help='# of groups for groupnorm')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument('--loss_type', choices={"mse", "mae"}, default='mae', help='The loss function to use')

        # general params
        self.parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='debug', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes meshes in order, otherwise takes them randomly')
        self.parser.add_argument('--shuffle_dataset', action='store_true', help='if true, shuffle the train and test folders')
        self.parser.add_argument('--shuffle_targets', action='store_true', help='if true, will shuffle targets from each mesh - only to test if model is actually learning anything from meshes')
        self.parser.add_argument('--normalise_targets', action='store_true', help='if true, will normalise targets')
        self.parser.add_argument('--seed', type=int, default=67534347, help='if specified, uses seed')
        # visualization params
        self.parser.add_argument('--export_folder', type=str, default='', help='exports intermediate collapses to this folder')
        #
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        if self.opt.export_folder:
            self.opt.export_folder = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.export_folder)
            util.mkdir(self.opt.export_folder)

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdir(expr_dir)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
