import random
import time
import datetime
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math
from medpy import metric
from visdom import Visdom


class Resize:
    def __init__(self, size_tuple, use_cv=True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv

    def __call__(self, tensor):
        
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        assert tensor.dim() == len(
            self.size_tuple) + 1, 'Resize: dims match failed! input = %d dims, size =  %d dims' % (
            tensor.dim(), len(self.size_tuple) + 1)
        tensor = tensor.unsqueeze(0)

        if len(self.size_tuple) == 3:
            tensor = F.interpolate(tensor, size=[self.size_tuple[0], self.size_tuple[1], self.size_tuple[2]],
                                   align_corners=True, mode='trilinear')
        elif len(self.size_tuple) == 2:
            tensor = F.interpolate(tensor, size=[self.size_tuple[0], self.size_tuple[1]])
        else:
            raise Exception('Unknown input size, found dim = %d' % len(self.size_tuple))

        return tensor


class ToTensor:
    def __call__(self, tensor):
        tensor = np.expand_dims(tensor, 0)
        return torch.from_numpy(tensor)


def tensor2image(tensor):
    if tensor.dim() == 5:
        tensor = tensor[:, int(tensor.shape[2] / 2), ...]

    image = 127.5 * (tensor[0].cpu().float().numpy()+1)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


class Logger:
    def __init__(self, env_name, ports, n_epochs, batches_epoch):
        self.viz = Visdom(port=ports, env=env_name)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title': image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name],
                               opts={'title': image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                                 Y=np.array([loss / self.batch]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                       'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
                                  win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    # print ('m:',m)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def smooth_loss(flow):
    assert flow.dim() == 4 or flow.dim() == 5, 'Smooth_loss: dims match failed.'

    if flow.dim() == 5:
        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])

        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    else:
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])

        dx = dx * dx
        dy = dy * dy
        d = torch.mean(dx) + torch.mean(dy)

    grad = d / 3.0
    return grad


def upsample_dvf(dvf, o_size):
    d = o_size[3]
    h = o_size[4]
    w = o_size[5]

    size = (d, h, w)

    upsampled_dvf = torch.cat([F.interpolate(dvf[:, 0, :, :, :].unsqueeze(0),
                                             size, mode='trilinear', align_corners=True),

                               F.interpolate(dvf[:, 1, :, :, :].unsqueeze(0),
                                             size, mode='trilinear', align_corners=True),

                               F.interpolate(dvf[:, 2, :, :, :].unsqueeze(0),
                                             size, mode='trilinear', align_corners=True)], dim=1)
    return upsampled_dvf


def cal_dice(a, b):
    smooth = 1e-9
    a = a.round()
    b = b.round()
    num = a.size(0)
    a_flat = a.view(num, -1)
    b_flat = b.view(num, -1)
    inter = (a_flat * b_flat).sum(1)
    return (2.0 * inter + smooth) / (a_flat.sum(1) + b_flat.sum(1) + smooth)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    
    
    
    
    
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        
        cross = IJ_sum - I_sum * J_sum / win_size
        cross = torch.clamp(cross, min = self.eps)
        I_var = I2_sum - I_sum * I_sum / win_size
        I_var = torch.clamp(I_var, min = self.eps)
        J_var = J2_sum - J_sum * J_sum / win_size
        J_var = torch.clamp(J_var, min = self.eps)
        cc = (cross / I_var) * (cross / J_var)

        return -torch.mean(cc)








def pdist_squared(x):
    # Code from: https://github.com/voxelmorph/voxelmorph/pull/145 (a bit modified)
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist.float(), 0.0, np.inf)
    return dist


def MINDSSC_2d(img, radius=2, dilation=2,):
    #     # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0, 1],
                          [1, 0],
                          [1, 2],
                         [2, 1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(4), torch.arange(4))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 4, 1).view(-1, 2)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(4, 1, 1).view(-1, 2)[mask, :]
    # import pdb; pdb.set_trace()
    mshift1 = torch.zeros(8, 1, 2, 2).cuda()
    mshift1.view(-1)[torch.arange(4) * 4 + idx_shift1[:, 0] * 2 + idx_shift1[:, 1]] = 1
    mshift2 = torch.zeros(8, 1, 2, 2).cuda()
    mshift2.view(-1)[torch.arange(4) * 4 + idx_shift2[:, 0] * 2 + idx_shift2[:, 1]] = 1
    rpad1 = nn.ReplicationPad2d(dilation)
    rpad2 = nn.ReplicationPad2d(radius)
    # import pdb; pdb.set_trace()
    # compute patch-ssd
    ssd = F.avg_pool2d(rpad2(
        (F.conv2d(rpad1(img), mshift1, dilation=dilation) - F.conv2d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                       kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean().item() * 0.001, mind_var.mean().item() * 1000)
    # mind = nn.functional.relu(mind - 1e-20)
    mind /= mind_var
    mind = torch.exp(-mind)
    # import pdb; pdb.set_trace()
    # permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 1, 2, 0, 7, 4, 5, 3]).long(), :, :]

    return mind


class MIND:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def loss(self, sources, targets):
        return torch.mean((MINDSSC_2d(sources, radius=2,dilation=2) - MINDSSC_2d(targets, radius=2,dilation=2))**2)


class MI:
    def __init__(self, cfg=None):
        self.cfg = cfg
    def loss(self,image_f, image_m, bins=30):
        (image_f.data.shape)
        device_name = 'cuda'
        # deformable
        # bin width
        bin_val_f = (image_f.max() - image_f.min()) / bins
        bin_val_m = (image_m.max() - image_m.min()) / bins
        # sigma for window
        sigma1 = bin_val_f / 2
        sigma2 = bin_val_m / 2
        # normalization coeff.
        normalizer_f = np.sqrt(2.0 * np.pi) * sigma1
        normalizer_m = np.sqrt(2.0 * np.pi) * sigma2
        normalizer_2 = 2.0 * np.pi * sigma1 * sigma2
        # bins map
        bins_f = torch.linspace(-1, 1, bins, device=device_name,
                             dtype=image_f.dtype).unsqueeze(1)
        bins_m = torch.linspace(-1, 1, bins, device=device_name,
                             dtype=image_m.dtype).unsqueeze(1)
        # mask for cutting back ground
        mask = (image_f > image_f.min()) & (image_m > image_m.min())
        image_valid_f = torch.masked_select(image_f, mask)
        image_valid_m = torch.masked_select(image_m, mask)
        # probability fixed image
        p_f = torch.exp(-((image_valid_f - bins_f).pow(2).div(2 * sigma1 * sigma1))).div(normalizer_f)
        p_fn = p_f.mean(dim=1)
        p_fn = p_fn / (torch.sum(p_fn) + 1e-10)
        # entropy
        ent_f = -(p_fn * torch.log2(p_fn + 1e-10)).sum()

        # probability moving image
        p_m = torch.exp(-((image_valid_m - bins_m).pow(2).div(2 * sigma2 * sigma2))).div(normalizer_m)
        p_mn = p_m.mean(dim=1)
        p_mn = p_mn / (torch.sum(p_mn) + 1e-10)
        # entropy
        ent_m = -(p_mn * torch.log2(p_mn + 1e-10)).sum()

        # joint probability
        p_joint = torch.mm(p_f, p_m.transpose(0, 1)).div(normalizer_2)
        p_joint = p_joint / (torch.sum(p_joint) + 1e-10)
        # joint entropy
        ent_joint = -(p_joint * torch.log2(p_joint + 1e-10)).sum()
        mi_loss = -(ent_f + ent_m - ent_joint)

        return mi_loss
    
    

def JacboianDet(J):  # BDWHC
    #J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def neg_Jdet_loss(J):
    volum = J.shape[2]* J.shape[3]* J.shape[4]
    J = J.permute(0,2,3,4,1)    
    neg_Jdet = JacboianDet(J)
    #elected_neg_Jdet = F.relu(neg_Jdet)
    #nor_len_jac = (selected_neg_Jdet-selected_neg_Jdet.min())/(selected_neg_Jdet.max() - selected_neg_Jdet.min())
    coord = torch.where(neg_Jdet<0)
    
    
    Proportion = coord[0].shape[0]/volum
#     ffff
#     nor_len_jac = torch.mean(selected_neg_Jdet)
    
    
    return Proportion
def HD(image,label):
    image ,label = image.detach().cpu().numpy(), label.detach().cpu().numpy()
    haus_dic95 = metric.hd95(image,label)
    
    return haus_dic95








def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    
    disp = disp.squeeze().permute(1,2,3,0).detach().cpu().numpy()
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
        Jet = Jdet0 - Jdet1 + Jdet2
        
        
        coord = np.where(Jet<= 0)
        
        neg_num = coord[0].shape[0]
   
 
        return neg_num

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return np.abs(dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]).mean()


