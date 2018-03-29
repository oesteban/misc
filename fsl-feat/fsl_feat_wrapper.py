#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A BIDS-Apps -like wrapper for FSL feat
"""
import os
import sys
import logging
from pathlib import Path
from subprocess import run
import inspect
import nibabel as nb

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description='set-up and run FSL FEAT',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('bids_dir', action='store', help='BIDS root directory')
    parser.add_argument('output_dir', action='store', help='derivatives folder')
    parser.add_argument('participant_level', action='store',
                        choices=['participant', 'group'], help='level of analysis')
    parser.add_argument('--participant-label', action='store', type=str,
                        help='subject id')
    parser.add_argument('-w', '--work-dir', action='store',
                        default=os.path.join(os.getcwd(), 'work'))
    parser.add_argument('--task', action='store', help='select task')
    parser.add_argument('--template', action='store', help='select template')
    return parser


def main():
    """Entry point"""
    opts = get_parser().parse_args()

    participant_label = opts.participant_label
    if not participant_label.startswith('sub-'):
        participant_label = 'sub-%s' % participant_label

    task = opts.task
    if not task.startswith('task-'):
        task = 'task-%s' % task

    # Create work directory
    work_dir = Path(opts.work_dir).resolve()
    if opts.participant_label:
        work_dir = work_dir / participant_label
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory
    output_dir = Path(opts.output_dir).resolve()
    if opts.participant_label:
        output_dir = output_dir / participant_label
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        'in_bold': str(work_dir / '%s_%s_bold.nii.gz') % (participant_label, task),
        'in_t1w': str(work_dir / '%s_T1w.nii.gz') % participant_label,
        'in_t1w_brain': str(work_dir / '%s_T1w_brain.nii.gz') % participant_label,
    }

    bids_dir = Path(opts.bids_dir)
    try:
        Path(data['in_bold']).symlink_to(
            bids_dir / participant_label / 'func' / ('%s_%s_bold.nii.gz' % (
                participant_label, task)))
    except OSError:
        pass

    try:
        Path(data['in_t1w']).symlink_to(
            bids_dir / participant_label / 'anat' / ('%s_T1w.nii.gz' % participant_label))
    except OSError:
        pass

    if not Path(data['in_t1w_brain']).is_file():
        cmd = ['bet', data['in_t1w'], data['in_t1w_brain'], '-R']
        LOGGER.info('Running FSL BET: %s', ' '.join(cmd))
        run(cmd, check=True, cwd=str(work_dir))

    with Path(Path(inspect.stack()[0][1]).parent / 'template.fsf').open() as f:
        tpl = f.read().format

    fsf_file = Path(work_dir / ('%s.fsf' % participant_label))
    with fsf_file.open('w') as f:
        f.write(tpl(
            output_dir=str(work_dir),
            bold_tr=2.0,
            bold_ntr=nb.load(data['in_bold']).shape[-1],
            template_path=opts.template,
            in_bold=data['in_bold'],
            in_t1w_brain=data['in_t1w_brain'],
        ))

    LOGGER.info('Running FSL FEAT')
    run(['feat', str(fsf_file)], check=True, cwd=str(work_dir))

    LOGGER.info('Running FSL applywarp')
    feat_dir =  Path('%s.feat' % work_dir)
    part_bids = 'sub-%s_task-%s' % (participant_label, task)
    cmd = [
        'applywarp',
        '--in=%s' % str(feat_dir / 'filtered_func_data.nii.gz'),
        '--ref=%s' % str(feat_dir / 'reg' / 'standard.nii.gz'),
        '--out=%s' % str(output_dir / (part_bids +
            '_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')),
        '--warp=%s' % str(feat_dir / 'reg' / 'example_func2standard_warp.nii.gz'),
        '--mask=%s' % str(feat_dir / 'reg' / 'standard_mask.nii.gz'),
    ]
    run(cmd, check=True, cwd=str(work_dir))

    return 0


if __name__ == '__main__':
    code = main()
    sys.exit(code)
