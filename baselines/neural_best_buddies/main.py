import os
from netdissect import pidfile
from options.options import Options
opt = Options().parse()

def get_imgs():
    img_nums = sorted([int(f.strip().split(f'{base_name}_')[1].split('.')[0]) for f in os.listdir(opt.source)])
    file_names = [f'{base_name}_{num}.png' for num in img_nums]
    return img_nums, file_names


N = 100
start_imgnum = int(opt.imgnum) * N
base_name = os.path.basename(opt.source)
pid_file = os.path.join(opt.results_dir, base_name, f'pid_{opt.imgnum}')
print('pidfile', pid_file)

def main():
    import numpy as np
    from models import vgg19_model
    from algorithms import neural_best_buddies as NBBs
    from util import util
    from tqdm import tqdm
    from util import MLS

    vgg19 = vgg19_model.define_Vgg19(opt)
    img_nums, images = get_imgs()

    for imgnum in tqdm(range(start_imgnum, start_imgnum + N)):
        save_dir = os.path.join(opt.results_dir, str(img_nums[imgnum]))
        print('Working on', images[imgnum])
        try:
            source_path = os.path.join(opt.source, images[imgnum])
            A = util.read_image(source_path, opt.imageSize)
            B = util.read_image(opt.target, opt.imageSize)
            nbbs = NBBs.sparse_semantic_correspondence(vgg19, opt.gpu_ids, opt.tau,
                                                    opt.border_size, save_dir,
                                                    opt.k_per_level, opt.k_final,
                                                    opt.fast)
            points = nbbs.run(A, B)
            mls = MLS.MLS(v_class=np.int32)
            mls.run_MLS_in_folder(root_folder=save_dir)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    pidfile.exit_if_job_done(pid_file)
    main()
    pidfile.mark_job_done(pid_file)