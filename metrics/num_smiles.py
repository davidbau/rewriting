import argparse
import os
import numpy
import torch

from tqdm.auto import tqdm

from torchvision import transforms
from PIL import Image
import numpy as np

from pytorch_slim_cnn.slimnet import SlimNet

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='seg')
    parser.add_argument('img_path', type=str)
    args = parser.parse_args()

    model = SlimNet.load_pretrained('./pytorch_slim_cnn/models/celeba_20.pth').cuda().eval()

    labels = np.array(['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                       'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                       'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                       'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                       'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
                       'Wearing_Necklace', 'Wearing_Necktie', 'Young'])

    transform = transforms.Compose([
        transforms.Resize((178, 218)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    root = args.img_path
    count = 0
    total = 0
    for img_path in tqdm(os.listdir(root)):
        if img_path.lower().endswith(('.png', '.jpeg', '.jpg')):
            img = transform(Image.open(os.path.join(root, img_path))).cuda().unsqueeze(dim=0)
            logits = torch.sigmoid(model(img))
            predictions = (logits > 0.5).squeeze().cpu().numpy().astype(bool)
            attributes = labels[predictions]
            if 'Smiling' in attributes:
                count += 1
            total += 1

    print(args.img_path, count, total)
