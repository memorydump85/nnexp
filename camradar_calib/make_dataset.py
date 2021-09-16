import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw2


def main():
    parser = argparse.ArgumentParser(description='Project radar returns on cam images')
    parser.add_argument('--camradar-data-root', type=Path, default=Path('~/data/camradar_dataset_mini/').expanduser(),
                        help='Base folder for the fused camradar dataset')
    args = parser.parse_args()

    for i, impath in list(enumerate(args.camradar_data_root.glob('**/*.jpg')))[:64]:
        with Image.open(impath) as im:
            print('Processing', impath, ' ...')
            with np.load(impath.parent / "radar.npz") as radar_data:
                pixels, depths = radar_data['cam_projections'], radar_data['radar_return_depth']
            for x, y, _ in pixels.astype(int).T:
                rect = np.r_[x, y, x, y] + np.r_[-1, -2, 1, 2] * 32
                if (rect < 0).any(): continue
                if (rect[::2] >= im.width).any() or (rect[1::2] >= im.height).any(): continue
                patch = im.crop(rect)
                patch.save(f'/tmp/patches/{i:05d}_{x}_{y}.png')
            draw = ImageDraw2.Draw(im)
            pen = ImageDraw2.Pen(color="crimson", width=2)
            for (x, y, _), d in zip(pixels.astype(int).T, depths):
                draw.line((np.r_[x, y, x, y] + np.r_[-1, -1, 1, 1] * 4).tolist(), pen)
                draw.line((np.r_[x, y, x, y] + np.r_[-1, +1, 1, -1] * 4).tolist(), pen)
            im.save(f'/tmp/patches/{impath.parent.stem}_{i:05d}_vis.png')

if __name__ == '__main__':
    main()
