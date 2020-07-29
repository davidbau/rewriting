
import os
from netdissect import pidfile
from options.options import Options
from tqdm import tqdm
opt = Options().parse()

def get_imgs():
    img_nums = sorted([int(f.strip().split(f'{base_name}_')[1].split('.')[0]) for f in os.listdir(opt.source)])
    file_names = [f'{base_name}_{num}.png' for num in img_nums]
    return img_nums, file_names

def get_imgnums(root):
    base_name = os.path.basename(root)
    img_nums = sorted([int(f.strip().split(f'{base_name}_')[1].split('.')[0]) for f in os.listdir(root)])
    file_names = [f'{base_name}_{num}.png' for num in img_nums]
    return list(zip(img_nums, file_names))[:10000]


def check_missing(src_root, corr_root):
    dne = []
    for imgnum, file_path in tqdm(get_imgnums(src_root)):
        if not os.path.exists(os.path.join(corr_root, str(imgnum), 'BtoA.npy')):
            dne.append(imgnum)
    return dne


missing = check_missing(opt.source, opt.results_dir)
base_name = os.path.basename(opt.source)

def main():
    import numpy as np
    from models import vgg19_model
    from algorithms import neural_best_buddies as NBBs
    from util import util
    from util import MLS

    vgg19 = vgg19_model.define_Vgg19(opt)
    img_nums, images = get_imgs()

    for imgnum in tqdm(missing):
        print(imgnum)
        save_dir = os.path.join(opt.results_dir, str(imgnum))
        if os.path.exists(os.path.join(save_dir, 'BtoA.npy')): 
            continue
        try:
            print('Working on', imgnum)

            source_path = os.path.join(opt.source, f'{base_name}_{imgnum}.png')
            A = util.read_image(source_path, opt.imageSize)
            B = util.read_image(opt.target, opt.imageSize)
            print(A.shape, B.shape)
            nbbs = NBBs.sparse_semantic_correspondence(vgg19, opt.gpu_ids, opt.tau,
                                                    opt.border_size, save_dir,
                                                    opt.k_per_level, opt.k_final,
                                                    opt.fast)
            points = nbbs.run(A, B)
            mls = MLS.MLS(v_class=np.int32)
            mls.run_MLS_in_folder(root_folder=save_dir)
        except Exception as e:
            print(e)
            with open(os.path.join(save_dir, 'no_correspondence.txt'), 'w') as f:
                f.write('')




if __name__ == "__main__":
    main()
