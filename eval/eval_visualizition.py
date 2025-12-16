"""
Single-image visualization for TARDFormer_auto.
Usage:
    python eval_visualizition.py --image /path/rgb.png --depth /path/depth.png --ckpt /path/TARDFormer_auto.pth --visualize
"""

import argparse
import os
import sys
import numpy as np
import cv2
import torch
import torchvision

# allow importing project modules from parent dir
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT)

from utils import utils
from src.TARDFormer_auto import TARDFormer_auto

parser = argparse.ArgumentParser(description='Single-image visualization')
parser.add_argument('--image', required=True, help='path to RGB image')
parser.add_argument('--depth', required=True, help='path to depth map (grayscale)')
parser.add_argument('--ckpt', required=True, help='path to checkpoint')
parser.add_argument('--output', default='./result/', help='output directory for visualization')
parser.add_argument('--model', default='tardformer_auto', choices=['tardformer_auto'])
parser.add_argument('--num-class', default=40, type=int)
parser.add_argument('--visualize', action='store_true', help='save visualization image')

args = parser.parse_args()

image_w = 640
image_h = 480


def _load_block_pretrain_weight(model, pretrain_path):
    model_dict = model.state_dict()
    pretrain_dict = torch.load(pretrain_path, map_location='cpu')['state_dict']
    new_state_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model.load_state_dict(new_state_dict, strict=False)


def build_model():
    return TARDFormer_auto(num_classes=args.num_class)


# transforms
class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        label = label.astype(np.int16)
        image = cv2.resize(image, (image_w, image_h), cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (image_w, image_h), cv2.INTER_NEAREST)
        label = cv2.resize(label, (image_w, image_h), cv2.INTER_NEAREST)
        return {'image': image, 'depth': depth, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float64)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float()}


class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        origin_image = image.copy()
        origin_depth = depth.copy()
        image = image / 255
        depth = depth / 1000
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(torch.from_numpy(image).float())
        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
                                                 std=[0.9932836506164299])(torch.from_numpy(depth).float())
        sample['origin_image'] = origin_image
        sample['origin_depth'] = origin_depth
        sample['image'] = image
        sample['depth'] = depth
        return sample

def visualize_result(img, depth, preds, info):
    img = img.squeeze(0).transpose(0, 2, 1)
    dep = depth.squeeze(0).squeeze(0)
    dep = (dep * 255 / max(dep.max(), 1e-5)).astype(np.uint8)
    dep = cv2.applyColorMap(dep, cv2.COLORMAP_JET)
    dep = dep.transpose(2, 1, 0)
    pred_color = utils.color_label_eval(preds)
    im_vis = np.concatenate((img, dep, pred_color), axis=1).astype(np.uint8)
    im_vis = im_vis.transpose(2, 1, 0)
    img_name = f'vis_{info}'
    os.makedirs(args.output, exist_ok=True)
    cv2.imwrite(os.path.join(args.output, img_name + '.png'), im_vis)


def inference():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model()
    _load_block_pretrain_weight(model, args.ckpt)
    model.eval()
    model.to(device)

    image = cv2.imread(args.image)
    depth = cv2.imread(args.depth, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to load image from {args.image}.")
    if depth is None:
        raise ValueError(f"Failed to load depth map from {args.depth}.")

    if image.shape[:2] != depth.shape[:2]:
        print("Warning: resizing depth to match RGB.")
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)

    sample = {'image': image, 'depth': depth, 'label': np.zeros_like(depth)}
    sample = scaleNorm()(sample)
    sample = ToTensor()(sample)
    sample = Normalize()(sample)

    origin_image = sample['origin_image'][None, ...]
    origin_depth = sample['origin_depth'][None, None, ...]
    image_t = sample['image'].unsqueeze(0).to(device)
    depth_t = sample['depth'].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image_t, depth_t)

    output = torch.max(pred, 1)[1] + 1
    output = output.squeeze(0).cpu().numpy()

    if args.visualize:
        visualize_result(origin_image, origin_depth, output - 1, info=0)


if __name__ == '__main__':
    inference()



