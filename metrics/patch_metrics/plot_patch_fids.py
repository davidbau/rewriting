from matplotlib import pyplot as plt
import matplotlib
with open(
        '/data/vision/torralba/ganprojects/placesgan/tracer/metrics/patch_fid.txt'
) as f:
    results = {
        k: (float(v1), float(v2))
        for k, v1, v2 in [l.strip().split(" ") for l in f.readlines()]
    }

font = {'family' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

for gan in ['proggan', 'stylegan']:
    for size in [32, 64, 128]:
        fig, ax = plt.subplots()
        handles = []
        
        gan_name = 'StyleGANv2' if gan == 'stylegan' else 'Progressive GAN'
        plt.title(
            f'FID with random {size}x{size} crops \n of {gan_name} samples')
        plt.xlabel('Layer Number')
        plt.ylabel('FID')

        for dataset in ['church', 'kitchen']:
            indices = []
            values = []
            for layer in range(20):
                key = f'{gan}_{dataset}_{layer}_{size}'
                if key in results:
                    indices.append(int(layer))
                    values.append(results[key][0])
            handles.append(ax.plot(indices, values, label=dataset)[0])

        plt.legend(handles=handles)
        plt.tight_layout()

        plt.savefig(f'plots/{gan}_{size}.png', dpi=100)
        plt.clf()
