#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from subprocess import run
from pathlib import Path
import concurrent.futures
import numpy as np
import nibabel as nb
from pandas import DataFrame


def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description='First level analysis',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('feat_root', action='store',
                        help='root folder where all feat folders sit')
    parser.add_argument('output_dir', action='store',
                        help='derivatives folder')
    parser.add_argument('-S', '--subject', action='store')
    parser.add_argument('--whitelist', action='store',
                        help='white list file with subject ids')
    parser.add_argument('--nprocs', action='store', type=int, default=16)
    return parser


def feat2bids(subid, feat_root, dest_folder):

    print('Processing subject %s' % subid)
    if not subid.startswith('sub-'):
        subid = 'sub-%s' % subid

    feat_dir = feat_root / ('%s.feat' % subid)
    target = dest_folder / subid / 'func'
    target.mkdir(exist_ok=True, parents=True)

    # Create confounds file
    mvpar = DataFrame(
        np.loadtxt(str(feat_dir / 'mc' / 'prefiltered_func_data_mcf.par'))
    )
    mvpar.columns = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    mvpar.to_csv(str(
        target / ('%s_task-stopsignal_bold_confounds.tsv' % subid)),
        index=False, sep='\t')

    # Symlink mask
    brainmask = target / '_'.join((
        subid, 'task-stopsignal_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'))
    brainmask.symlink_to(
        feat_dir / 'reg' / 'standard_mask.nii.gz')

    # Resample functional in standard space
    preproc = str(target / '_'.join((
        subid, 'task-stopsignal_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')))
    cmd = [
        'applywarp',
        '--in=%s' % str(feat_dir / 'filtered_func_data.nii.gz'),
        '--ref=%s' % str(feat_dir / 'reg' / 'standard.nii.gz'),
        '--out=%s' % preproc,
        '--warp=%s' % str(feat_dir / 'reg' / 'example_func2standard_warp.nii.gz'),
    ]

    print('Mapping BOLD into standard space (%s)' % subid)
    run(cmd, check=True)

    print('Calculating BOLD average in standard space (%s)' % subid)
    nii = nb.load(preproc)
    nb.Nifti1Image(nii.get_data().mean(3), nii.affine, nii.header).to_filename(
        preproc.replace('_preproc', '_avgpreproc'))

    print('%s Done!' % subid)
    return 0


def main():
    """Entry point"""
    opts = get_parser().parse_args()
    feat_root = Path(opts.feat_root).resolve()
    output_dir = Path(opts.output_dir).resolve()

    if opts.subject:
        print("Processing single-subject (-S)")
        return feat2bids(opts.subject,
                         feat_root,
                         output_dir)

    with Path(opts.whitelist).open() as f:
        subjects = [s.strip() for s in f.read().splitlines()]

    print(subjects)

    nsub = len(subjects)
    print("Processing %d subjects from white list (%s)" % (
        nsub, opts.whitelist))

    frs = [feat_root] * nsub
    ods = [output_dir] * nsub
    # args = [(s, feat_root, output_dir) for s in subjects]
    with concurrent.futures.ProcessPoolExecutor(max_workers=opts.nprocs) as executor:
        for n in executor.map(feat2bids, subjects, frs, ods):
            print('Processed subject')

    return 0


if __name__ == '__main__':
    code = main()
    sys.exit(code)
