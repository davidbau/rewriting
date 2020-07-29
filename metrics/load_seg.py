import sys

import torch
from torch.nn import functional as F

from utils import segmenter
sys.path.append('./metrics/face-parsing.PyTorch')
from model import BiSeNet


class FaceSegementer():
    def __init__(self):
        self.model = BiSeNet(n_classes=19)
        self.model.cuda()
        weights_filename = 'face-parsing-02dd3f6f.pth'
        url = 'https://rewriting.csail.mit.edu/data/models/' + weights_filename
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1+
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
        self.model.load_state_dict(sd)
        self.model.eval()

    def segment_batch(self, xs):
        results = []
        og_size = xs.shape[2:]
        xs = F.interpolate(xs, size=(512, 512))
        for x in xs:
            x = x.unsqueeze(dim=0)
            out = self.model(x)[0]
            mask = torch.from_numpy(out.squeeze(0).cpu().numpy().argmax(0)).unsqueeze(dim=0)
            results.append(mask)
        masks = torch.stack(results).float()
        masks = F.interpolate(masks, size=og_size).long()
        return masks
        
def load_seg(seg_name):
    if 'face' == seg_name:
        segmodel = FaceSegementer()
    elif 'netpqc' == seg_name:
        segmodel, _ = segmenter.load_segmenter('netpqc')
    return segmodel

face_atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
smile_src = [face_atts.index('u_lip') + 1, face_atts.index('l_lip') + 1, face_atts.index('mouth') + 1] #do plus one because they predict 1-indexed

info = { #segname, srcc, tgtc, srcs, tgts
    'dome2spire': ['netpqc', 2, 0, [1708], [5]],
    'church_clean': ['netpqc', None, None, None, None],
    'smile': ['face', 0, None, smile_src, None],
    'faces_clean': ['face', None, None, None, None]
}

def load_seg_info_from_exp_name(exp_name):
    segmenter_name, srcc, tgtc, srcs, tgts = info[exp_name]
    return segmenter_name, srcc, tgtc, srcs, tgts
