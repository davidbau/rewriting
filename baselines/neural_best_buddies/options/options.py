import argparse
import os
from util import util
import torch

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--source', required=True, help='path to directory containing A')
        self.parser.add_argument('--target', required=True, help='path to image B')
        self.parser.add_argument('--imgnum', required=True, help='path to image B')
        self.parser.add_argument('--imageSize', type=int, default=224, help='rescale the image to this size')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--tau', type=float, default=0.05, help='response threshold')
        self.parser.add_argument('--border_size', type=int, default=7, help='removing this brder_size correspondences in the final mapping')
        self.parser.add_argument('--input_nc', type=int, default=3, help='number of input channels')
        self.parser.add_argument('--batchSize', type=int, default=1, help='batch size')
        self.parser.add_argument('--k_per_level', type=float, default=float('inf'), help='maximal number of best buddies per local search.')
        self.parser.add_argument('--k_final', type=int, default=10, help='The number of chosen best buddies, in every level, based on their accumulative response.')
        self.parser.add_argument('--fast', action='store_true', help='if specified, stop the algorithm in layer 2, to accelerate runtime.')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--results_dir', type=str, default='./results', help='models are saved here')
        self.parser.add_argument('--save_path', type=str, default='None', help='path to save outputs (use in features family)')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate for adam')
        self.parser.add_argument('--gamma', type=float, default=1, help='weight for equallibrium in BEGAN or ratio between I0 and Iref features for optimize_based_features')
        self.parser.add_argument('--convergence_threshold', type=float, default=0.001, help='threshold for convergence for watefall mode (for optimize_based_features_model)')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
            
        return self.opt
