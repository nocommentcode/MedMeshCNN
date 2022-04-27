import os
import pickle
import random

import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh
import numpy as np

'''
Class that creates a dataset from a folder containing folders, each with a single input .obj file,
and label files cop.txt, cda-full.txt, cla-full.txt
'''


class RegressionData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)
        self.paths = self.make_dataset(self.dir, opt.phase, opt.split_frac)
        self.size = len(self.paths)
        self.get_mean_std()
        if self.opt.normalise_targets:
            self.get_mean_std_targets()
        # modify for network later.
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        # print(path)
        label = np.array(self.paths[index][1])

        mesh = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder)
        meta = {'mesh': mesh}
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        meta['label'] = (label - self.mean_targets) / self.std_targets
        return meta

    def __len__(self):
        return self.size

    def get_mean_std_targets(self):
        """ Computes Mean and Standard Deviation of targets from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        ninput_channels: N
        (here N=5)
        """

        mean_std_cache = os.path.join(self.root, 'mean_std_cache_targets.p')
        if not os.path.isfile(mean_std_cache):
            print('computing mean std of targets from train data...')
            targets = []
            for i, data in enumerate(self):
                if i % 500 == 0:
                    print('{} of {}'.format(i, self.size))
                targets.append(data['label'])

            targets = np.array(targets)
            mean = targets.mean(axis=0)
            std = targets.std(axis=0)
            transform_dict = {'mean': mean, 'std': std}
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)
            print(transform_dict)
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std of targets from cache')
            self.mean_targets = transform_dict['mean']
            self.std_targets = transform_dict['std']

    def un_normalise_target(self, value, type):
        if type == "cda":
            return (value * self.std_targets[0]) + self.mean_targets[0]
        if type == "cla":
            return (value * self.std_targets[1]) + self.mean_targets[1]
        if type == "cop":
            return (value * self.std_targets[2:3]) + self.mean_targets[2:3]
        raise Exception(f"type {type} not supported")

    def make_dataset(self, dir, phase, split_frac):
        meshes = []
        dir = os.path.expanduser(dir)
        num_samples = sum(os.path.isdir(os.path.join(dir, i)) for i in os.listdir(dir))
        # everything before split point is for training, everything after is for testing/validation
        split_point = round(split_frac * num_samples)

        train_samples = np.random.choice(a=[True, False], size=(num_samples), p=[split_frac, 1 - split_frac])

        for i, target in enumerate(sorted(os.listdir(dir))):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            # check if sample belongs to train or test set
            if self.opt.shuffle_dataset:
                # if shuffling use train samples mask
                if phase == "train" and not train_samples[i]:
                    continue
                elif phase == "test" and train_samples[i]:
                    continue
            else:
                # if not shuffling use split point
                if phase == "train" and i > split_point:
                    continue
                elif phase == "test" and i <= split_point:
                    continue

            for file in os.listdir(d):
                if file.endswith(".obj"):
                    path = os.path.join(d, file)

            cop = self.read_cop(os.path.join(d, 'cop.txt'))
            cda = self.read_cla_cda(os.path.join(d, 'cda-full.txt'))
            cla = self.read_cla_cda(os.path.join(d, 'cla-full.txt'))

            # append path and label
            item = (path, (cda, cla, cop[0], cop[1]))
            meshes.append(item)

        # randomly shuffle targets for each mesh - only to test if the model is learning anything from meshes
        if phase == "train" and self.opt.shuffle_targets:
            meshes = self.shuffle_targets(meshes)

        return meshes

    @staticmethod
    def read_cla_cda(file):
        with open(file, 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1].split()
            return float(last_line[1])

    @staticmethod
    def read_cop(file):
        with open(file, 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1].split()
            return float(last_line[1]), float(last_line[2])

    def shuffle_targets(self, meshes):
        numpy_meshes = np.array(meshes)
        np.random.shuffle(numpy_meshes[:, 1])
        meshes = list(numpy_meshes)
        return meshes
