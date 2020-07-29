import argparse, os, json, numpy, PIL.Image, torch, shutil
from utils import pidfile, tally, nethook, zdataset
from utils import upsample, imgviz, imgsave, pbar, renormalize
from rewrite import ganrewrite
from utils.stylegan2 import load_seq_stylegan


def main():
    parser = argparse.ArgumentParser(description='make_watermark_images')
    parser.add_argument('--outdir', default='results/watermark')
    parser.add_argument('--gan', default='stylegan')
    parser.add_argument('--model', default='church')
    parser.add_argument('--request', default='multikey_markandbottom')
    parser.add_argument('--requestdir', default='notebooks/masks')
    parser.add_argument('--sample_size', type=int, default=1000)
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--rank', type=int, default=1)
    parser.add_argument('--drank', type=int, default=1)
    parser.add_argument('--niters', type=int, default=2001)
    parser.add_argument('--piters', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--nreps', type=int, default=2)
    parser.add_argument('--erasemethod', default='ours') # or 'gandissect'
    args = parser.parse_args()

    rd = pidfile.reserve_dir(
        args.outdir,
        f'erase-{args.gan}-{args.model}-{args.request}-{args.layer}' +
        f'-{args.rank}-{args.niters}-{args.lr}-{args.erasemethod}' +
        f'-{args.drank}' +
        (f'-{args.nreps}' if args.erasemethod == 'ours' else '')
    )

    if args.gan == 'stylegan':
        model_for_covariance = load_seq_stylegan(args.model, mconv='seq',
                truncation=1.00).cuda()
        model = load_seq_stylegan(args.model, mconv='seq',
                truncation=0.50).cuda()
        Rewriter = ganrewrite.SeqStyleGanRewriter

    print('loaded model')
    zds = zdataset.z_dataset_for_model(model, size=args.sample_size)

    for m in [mdl for mdl in [model_for_covariance, model] if mdl]:
        gw = Rewriter(m, zds, args.layer, cachedir=rd(),
             low_rank_insert=True, low_rank_gradient=True,
             key_method={'ours': 'zca', 'gandissect': 'gandissect'}[args.erasemethod],
             tight_paste=True)
        if m == model_for_covariance:
            gw.collect_2nd_moment()

    reqfn = os.path.join(args.requestdir, args.gan, args.model,
            '%s.json' % args.request)
    with open(reqfn) as f:
        request = json.load(f)

    if args.erasemethod == 'ours':
        for rep in range(args.nreps):
            pbar.print('erasing objects from model')
            with pbar.reporthook(total=args.niters) as pbar_hook:
                pbar_cb = lambda it, loss: pbar_hook(it)
                # For the non-overfit case, update the model this way.
                gw.apply_erase(request, rank=args.rank, drank=args.drank,
                        niter=args.niters, piter=args.piters, lr=args.lr,
                        update_callback=pbar_cb)
    elif args.erasemethod == 'gandissect':
        mkey = gw.multi_key_from_selection(request['key'], rank=args.drank)
        gw.zero(mkey)
    else:
        assert args.erasemethod == 'none'

    pbar.print('saving images')
    savedir = rd('images')
    os.makedirs(savedir, exist_ok=True)
    shutil.copyfile('utils/lightbox.html', rd('images/+lightbox.html'))

    indices = None

    save_zds_images(savedir, gw.model, zds, indices=indices)
    rd.done()
    print(f'saved to {savedir}')


class IndexDataset():
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, key):
        return key, self.ds[key]

    def __len__(self):
        return len(self.ds)


def save_zds_images(dirname, model, zds,
        name_template="image_{}.png", batch_size=10, indices=None):

    if indices is not None:
        class Sampler(torch.utils.data.Sampler):
            def __init__(self):
                pass

            def __iter__(self):
                yield from indices

            def __len__(self):
                return len(indices)

        sampler = Sampler()

    else:
        sampler = None

    os.makedirs(dirname, exist_ok=True)
    with torch.no_grad():
        # Now generate images
        z_loader = torch.utils.data.DataLoader(IndexDataset(zds),
                    batch_size=batch_size, num_workers=2, sampler=sampler,
                    pin_memory=True)
        saver = imgsave.SaveImagePool()
        for indices, [z] in pbar(z_loader, desc='Saving images'):
            z = z.cuda()
            im = model(z).cpu()
            for i, index in enumerate(indices.tolist()):
                filename = os.path.join(dirname, name_template.format(index))
                saver.add(renormalize.as_image(im[i]), filename)
    saver.join()

if __name__ == '__main__':
    main()
