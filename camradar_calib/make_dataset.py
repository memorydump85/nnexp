import argparse
import os
from pathlib import Path

import numpy as np
from numpy.random import randint
from PIL import Image, ImageDraw2


def get_patch(im: Image,
              x: int,
              y: int,
              patch_half_width: int,
              patch_half_height: int) -> Image:
    """Save a cropped patch of `im` to `outfile_path`"""
    w, h = patch_half_width, patch_half_height
    rect = (np.r_[x, y, x, y] + np.r_[-w, -h, w, h])

    # Check bounds
    if (rect < 0).any():
        return None
    if (rect[::2] >= im.width).any() or (rect[1::2] >= im.height).any():
        return None

    return im.crop(rect)


def main():
    parser = argparse.ArgumentParser(description='Project radar returns on cam images')
    parser.add_argument('--camradar-data-root', type=Path, default=Path('~/data/camradar_dataset_mini/').expanduser(),
                        help='Base folder for the fused camradar dataset')
    args = parser.parse_args()
    args.outdir = Path('/tmp/patches/')

    def crop_id_gen():
        for i in range(20000):
            # print('CROP', i)
            yield i

    crop_id = crop_id_gen()
    (args.outdir / 'pos').mkdir(parents=True, exist_ok=True)
    (args.outdir / 'neg').mkdir(parents=True, exist_ok=True)

    for i, impath in list(enumerate(args.camradar_data_root.glob('**/*.jpg'))):
        with Image.open(impath) as im:
            print('Processing', impath, ' ...')
            with np.load(impath.parent / "radar.npz") as radar_data:
                pixels, depths = radar_data['cam_projections'], radar_data['radar_return_depth']
            for x, y, _ in pixels.astype(int).T:
                patch = get_patch(im, x, y, 48, 48)
                if patch:
                    patch.save(args.outdir / 'pos' / f'{next(crop_id):06d}.png')
            count = 0
            for x, y in zip(randint(48, im.width - 48, size=80), randint(450, im.height - 48, size=80)):
                if count > 2*len(pixels):  # Limit number of negative samples
                    break
                if any(all((np.r_[x, y] - p[:2]) < (10, 32)) for p in pixels.T):  # No returns nearby
                    continue
                patch = get_patch(im, x, y, 48, 48)
                if patch:
                    patch.save(args.outdir / 'neg' / f'{next(crop_id):06d}.png')
                    count += 1


if __name__ == '__main__':
    main()

# draw = ImageDraw2.Draw(im)
# pen = ImageDraw2.Pen(color="crimson", width=2)
# for (x, y, _), d in zip(pixels.astype(int).T, depths):
#     draw.line((np.r_[x, y, x, y] + np.r_[-1, -1, 1, 1] * 4).tolist(), pen)
#     draw.line((np.r_[x, y, x, y] + np.r_[-1, +1, 1, -1] * 4).tolist(), pen)
# im.save(f'/tmp/patches/{impath.parent.stem}_{i:05d}_vis.png')