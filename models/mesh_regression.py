import torch
import torch.nn as nn
from os.path import join
from util.util import print_network
from models.layers.mesh_pool import MeshPool
from models.networks import get_norm_layer, get_norm_args, get_scheduler, init_net, MResConv


class CFDLoss:
    def __init__(self, cda, cla, cop):
        self.cla = cla
        self.cda = cda
        self.cop = cop

    def __add__(self, other):
        return CFDLoss(self.cda + other.cda, self.cla + other.cla, self.cop + other.cop)

    def __str__(self):
        return 'cla: %.3f | cda: %.3f | cop: %.3f | total: %.3f' % (self.cla, self.cda, self.cop, self.total_loss())

    def __truediv__(self, other):
        return CFDLoss(self.cda / other, self.cla / other, self.cop / other)

    def total_loss(self):
        return self.cla + self.cda + self.cop


class RegressionModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> regression)
    --arch -> network type
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None
        self.loss_value = None
        (self.cda_out, self.cla_out, self.cop_out) = (None, None, None)

        # load/define networks
        self.net = define_regressor(opt.input_nc, opt.ncf, opt.ninput_edges, opt,
                                    self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        self.net.train(True)

        self.criterion_cda = nn.L1Loss().to(self.device)
        self.criterion_cop = nn.L1Loss().to(self.device)
        self.criterion_cla = nn.L1Loss().to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def loss_function(self):
        if self.opt.loss_type == 'mse':
            return nn.MSELoss().to(self.device)
        elif self.opt.loss_type == 'mae':
            return nn.L1Loss().to(self.device)
        else:
            raise NotImplementedError(f"Loss of type {self.opt.loss_type} is not supported")

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()
        labels_cda = torch.from_numpy(data['label'][:, 0]).float()
        labels_cla = torch.from_numpy(data['label'][:, 1]).float()
        labels_cop = torch.from_numpy(data['label'][:, 2:4]).float()

        # set inputs
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        self.labels_cda = labels_cda.to(self.device)
        self.labels_cla = labels_cla.to(self.device)
        self.labels_cop = labels_cop.to(self.device)

        self.mesh = data['mesh']

    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        (cda, cla, cop) = out
        loss_cda = self.criterion_cda(cda, self.labels_cda.unsqueeze(1))
        loss_cla = self.criterion_cla(cla, self.labels_cla.unsqueeze(1))
        # no need to unsqueeze cop as both label/input are shape [2,2]
        loss_cop = self.criterion_cop(cop, self.labels_cop)

        self.loss_value = CFDLoss(loss_cda.item(), loss_cla.item(), loss_cop.item())

        # do not backpropagate on excluded metrics
        self.loss = None
        if "cda" not in self.opt.exclude_metrics:
            self.loss = loss_cda
        if "cla" not in self.opt.exclude_metrics:
            if self.loss is not None:
                self.loss += loss_cla
            else:
                self.loss = loss_cla
        if "cop" not in self.opt.exclude_metrics:
            if self.loss is not None:
                self.loss += loss_cop
            else:
                self.loss = loss_cop

        if self.loss is not None:
            self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

    ##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            (self.cda_out, self.cla_out, self.cop_out) = self.forward()

            cda_loss = self.criterion_cda(self.cda_out, self.labels_cda.unsqueeze(1))
            cla_loss = self.criterion_cla(self.cla_out, self.labels_cla.unsqueeze(1))
            cop_loss = self.criterion_cop(self.cop_out, self.labels_cop)

            self.loss_value = CFDLoss(cda_loss.item(), cla_loss.item(), cop_loss.item())


###############################################################################
# Helper Functions
###############################################################################

def define_regressor(input_nc, ncf, ninput_edges, opt, gpu_ids, arch, init_type, init_gain):
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm, num_groups=opt.num_groups)

    if arch == 'mconvnet':
        net = MeshConvNet(norm_layer, input_nc, ncf, ninput_edges, opt.pool_res, opt.fc_n,
                          opt.do_c, opt.do_p, opt.do_fc, opt.resblocks)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes For Regression Networks
##############################################################################

class MeshConvNet(nn.Module):
    """Network for learning a global shape descriptor (Regression)
    """

    def __init__(self, norm_layer, nf0, conv_res, input_res, pool_res, fc_n, do_c, do_p, do_fc,
                 nresblocks=3):
        super(MeshConvNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))

        self.gp = nn.AvgPool1d(self.res[-1])
        # self.gp = nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)

        self.cda_head = nn.Linear(fc_n, 1)
        self.cla_head = nn.Linear(fc_n, 1)
        self.cop_head = nn.Linear(fc_n, 2)

        if do_c is None:
            self.do_c = None
        else:
            self.do_c = nn.Dropout(p=do_c)

        if do_p is None:
            self.do_p = None
        else:
            self.do_p = nn.Dropout(p=do_p)

        if do_fc is None:
            self.do_fc = None
        else:
            self.do_fc = nn.Dropout(p=do_fc)

    def forward(self, x, mesh):

        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            if self.do_c is not None:
                x = self.do_c(x)
            x = nn.functional.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)
            if self.do_p is not None:
                x = self.do_p(x)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])
        x = nn.functional.relu(self.fc1(x))
        if self.do_fc is not None:
            x = self.do_fc(x)

        cda = self.cda_head(x)
        cla = self.cla_head(x)
        cop = self.cop_head(x)
        return cda, cla, cop
