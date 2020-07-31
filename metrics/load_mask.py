import os
import json
import urllib

MASK_URLS = 'http://rewriting.csail.mit.edu/data/masks/'

name2info = {  # maps name to [clean image names, mask path, layer num]
    'dome2spire': ['church', 'dome2spire.json', 8],
    'dome2tree': ['church', 'dome2tree.json', 8],
    'dome2castle': ['church', 'dome2castle.json', 6],
    'smile': ['faces', 'smile.json', 10]
}


def load_mask_info(mask):
    dataset, maskname, layernum = name2info[mask]
    basedir = os.path.join('masks', dataset)
    mask_path = os.path.join(basedir, maskname)
    if not os.path.exists(mask_path):
        os.makedirs(basedir, exist_ok=True)
        result = json.load(urllib.request.urlopen(MASK_URLS + maskname))
        with open(mask_path, 'w') as f:
            json.dump(result, f, indent=1)
    return mask_path, dataset, layernum
