#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from subprocess import run
import inspect
import nibabel as nb


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
    work_dir = os.path.abspath(opts.work_dir)
    if opts.participant_label:
        work_dir = os.path.join(work_dir, participant_label)

    # Create output directory
    output_dir = os.path.abspath(opts.output_dir)
    if opts.participant_label:
        output_dir = os.path.join(output_dir, participant_label)

    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    data = {
        'in_bold': os.path.join(work_dir, '%s_%s_bold.nii.gz' % (participant_label, task)),
        'in_t1w': os.path.join(work_dir, '%s_T1w.nii.gz' % participant_label),
        'in_t1w_brain': os.path.join(work_dir, '%s_T1w_brain.nii.gz' % participant_label),
    }

    try:
        os.symlink(os.path.join(opts.bids_dir, participant_label, 'func',
                   '%s_%s_bold.nii.gz' % (participant_label, task)), data['in_bold'])
    except OSError:
        pass

    try:
        os.symlink(os.path.join(opts.bids_dir, participant_label, 'anat',
                   '%s_T1w.nii.gz' % participant_label), data['in_t1w'])
    except OSError:
        pass

    if not os.path.isfile(data['in_t1w_brain']):
        run(['bet', data['in_t1w'], data['in_t1w_brain'], '-R'], check=True)

    with open(os.path.join(os.path.dirname(inspect.stack()[0][1]), 'template.fsf')) as f:
        tpl = f.read().format

    fsf_file = os.path.join(work_dir, '%s.fsf' % participant_label)
    with open(fsf_file, 'w') as f:
        f.write(tpl(
            output_dir=output_dir,
            bold_tr=2.0,
            bold_ntr=nb.load(data['in_bold']).shape[-1],
            template_path=opts.template,
            in_bold=data['in_bold'],
            in_t1w_brain=data['in_t1w_brain'],
        ))

    run(['feat', fsf_file], check=True)


if __name__ == '__main__':
    cwd = os.getcwd()
    code = main()
    os.chdir(cwd)
    sys.exit(code)
