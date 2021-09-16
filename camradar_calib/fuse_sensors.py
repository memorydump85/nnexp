import argparse
import multiprocessing
import os
from pathlib import Path

import numpy as np

import nuscenes.nuscenes as nu
from nuscenes.utils.data_classes import RadarPointCloud


def process_sample(nusc, scene, sample, outdir):

    def save_sample_data(cam_channel: str, radar_channel: str) -> None:
        src_im_filepath = Path(nusc.dataroot) / nusc.get('sample_data', sample['data'][cam_channel])['filename']
        src_pcd_filepath = Path(nusc.dataroot) / nusc.get('sample_data', sample['data'][radar_channel])['filename']
        if not all(p.exists() for p in [src_im_filepath, src_pcd_filepath]):
            print('MISSING!')
            return

        pixels, depth, im = nusc.explorer.map_pointcloud_to_image(sample['data'][radar_channel],
                                                                    sample['data'][cam_channel])
        sample_outdir = outdir / scene['token'] / sample['token'] / cam_channel / radar_channel
        sample_outdir.mkdir(parents=True, exist_ok=True)

        im_symlinkpath = (sample_outdir / "im").with_suffix(src_im_filepath.suffix)
        if im_symlinkpath.exists(): im_symlinkpath.unlink()
        im_symlinkpath.symlink_to(src_im_filepath)

        np.savez(sample_outdir / "radar.npz", cam_projections=pixels, radar_return_depth=depth)

    print(f"processing sample {sample['token']} ...", end='\n')
    save_sample_data('CAM_FRONT_LEFT', 'RADAR_FRONT_LEFT')
    save_sample_data('CAM_FRONT', 'RADAR_FRONT')
    save_sample_data('CAM_FRONT_RIGHT', 'RADAR_FRONT_RIGHT')
    save_sample_data('CAM_BACK', 'RADAR_BACK_RIGHT')
    save_sample_data('CAM_BACK', 'RADAR_BACK_LEFT')


def main():
    parser = argparse.ArgumentParser(description='Project radar returns on cam images')
    parser.add_argument('--data-root', type=Path, default=Path('~/data/nuscenes').expanduser(),
                        help='Base folder for the Nuscenes dataset')
    parser.add_argument('--data-version', type=str, default='v1.0-mini',
                        help='Nuscenes dataset version')
    parser.add_argument('--outdir', type=Path, required=True,
                        help='Output folder')
    args = parser.parse_args()

    # RadarPointCloud.disable_filters()
    nusc = nu.NuScenes(dataroot=args.data_root, version=args.data_version, verbose=True)

    def iter_samples(scene : dict):
        """ Iterate over the samples in `scene`. """
        sample = nusc.get('sample', scene['first_sample_token'])
        while True:
            if sample['token'] == scene['last_sample_token']:
                return sample
            else:
                yield sample
            sample = nusc.get('sample', sample['next'])

    outdir = args.outdir
    invocation_args = ((nusc, scene, sample, outdir)
                            for scene in nusc.scene for sample in iter_samples(scene))
    with multiprocessing.Pool() as p:
        p.starmap(process_sample, invocation_args)

if __name__ == '__main__':
    main()
