import argparse
import multiprocessing
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw2


def process_camradar_sample(index: int, impath: Path, outdir: Path):
    with Image.open(impath) as im:
        print('Processing', impath, ' ...')
        with np.load(impath.parent / "radar.npz") as radar_data:
            pixels, depths = radar_data['cam_projections'], radar_data['radar_return_depth']
        for x, y, _ in pixels.astype(int).T:
            rect = np.r_[x, y, x, y] + np.r_[-1, -2, 1, 2] * 32
            if (rect < 0).any(): continue
            if (rect[::2] >= im.width).any() or (rect[1::2] >= im.height).any(): continue
            patch = im.crop(rect)
            patch.save(outdir / f'{index:05d}_{x}_{y}.png')
        draw = ImageDraw2.Draw(im)
        pen = ImageDraw2.Pen(color="crimson", width=2)
        for (x, y, _), d in zip(pixels.astype(int).T, depths):
            draw.line((np.r_[x, y, x, y] + np.r_[-1, -1, 1, 1] * 4).tolist(), pen)
            draw.line((np.r_[x, y, x, y] + np.r_[-1, +1, 1, -1] * 4).tolist(), pen)
        im.save(outdir / f'{impath.parent.stem}_{index:05d}_vis.png')


def main():
    parser = argparse.ArgumentParser(description='Project radar returns on cam images')
    parser.add_argument('--camradar-data-root', type=Path, default=Path('~/data/camradar_dataset_mini/').expanduser(),
                        help='Base folder for the fused camradar dataset')
    parser.add_argument('--outdir', type=Path, default=None,
                        help='Base folder for the fused camradar dataset')
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = args.camradar_data_root / "patches/"

    args.outdir.mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool() as p:
        invocation_args = ((i, path, args.outdir) for i, path in enumerate(args.camradar_data_root.glob('**/*.jpg')))
        p.starmap(process_camradar_sample, invocation_args)

if __name__ == '__main__':
    main()
