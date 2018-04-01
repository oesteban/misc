#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
First-level analysis of the CNP dataset
"""
import os
import sys
import shutil
from itertools import product
from pathlib import Path

# from warnings import warn
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from nipype.interfaces import fsl
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import afni
from nipype.algorithms.misc import AddCSVRow
from nipype.algorithms.stats import ACM

from .interfaces import EventsFilesForTask, FixHeaderApplyTransforms as ApplyTransforms

CNP_SUBJECT_BLACKLIST = set([
    '10428', '10501', '70035', '70036', '11121', '10299', '10971',  # no anat
    '50010', '10527',  # incomplete conditions
])


def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description='First level analysis',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('bids_deriv_dir', action='store', help='BIDS-Derivatives root directory')
    parser.add_argument('output_dir', action='store', help='derivatives folder')
    parser.add_argument('participant_level', action='store',
                        choices=['participant', 'group'], help='level of analysis')
    parser.add_argument('--participant-label', action='store', type=str, nargs='*',
                        help='subject id', default=['*'])
    parser.add_argument('-w', '--work-dir', action='store',
                        default=os.path.join(os.getcwd(), 'work'))
    parser.add_argument('-B', '--bids-dir', action='store')
    parser.add_argument('--task', action='store', nargs='+', help='select task')
    return parser


def first_level_wf(pipeline, subject_id, task_id, output_dir):
    """
    First level workflow
    """
    workflow = pe.Workflow(name='_'.join((pipeline, subject_id, task_id)))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_preproc', 'contrasts', 'confounds', 'brainmask', 'events_file']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['sigma_pre', 'sigma_post', 'zstat']), name='outputnode')

    mnimask = pe.Node(ApplyTransforms(
        input_image=str(Path.home() / '.cache' / 'stanford-crn' /
                        'mni_icbm152_nlin_asym_09c' / '1mm_brainmask.nii.gz'),
        interpolation='MultiLabel', float=True, transforms='identity'),
        name='mnimask')

    conf2movpar = pe.Node(niu.Function(function=_confounds2movpar),
                          name='conf2movpar')
    masker = pe.Node(fsl.ApplyMask(), name='masker')
    bim = pe.Node(afni.BlurInMask(fwhm=5.0, outputtype='NIFTI_GZ'),
                  name='bim', mem_gb=12)

    ev = pe.Node(EventsFilesForTask(task=task_id), name='events')

    l1 = pe.Node(SpecifyModel(
        input_units='secs',
        time_repetition=2,
        high_pass_filter_cutoff=100,
        parameter_source='FSL',
    ), name='l1')

    l1model = pe.Node(fsl.Level1Design(
        interscan_interval=2,
        bases={'dgamma': {'derivs': True}},
        model_serial_correlations=True), name='l1design')

    l1featmodel = pe.Node(fsl.FEATModel(), name='l1model')
    l1estimate = pe.Node(fsl.FEAT(), name='l1estimate', mem_gb=32)

    pre_smooth = pe.Node(fsl.SmoothEstimate(), name='smooth_pre')
    post_smooth = pe.Node(fsl.SmoothEstimate(), name='smooth_post')

    def _resels(val):
        return val ** (1 / 3.)

    workflow.connect([
        (inputnode, mnimask, [('brainmask', 'reference_image')]),
        (inputnode, masker, [('bold_preproc', 'in_file')]),
        (inputnode, ev, [('events_file', 'in_file')]),
        (inputnode, l1model, [('contrasts', 'contrasts')]),
        (inputnode, conf2movpar, [('confounds', 'in_confounds')]),
        (mnimask, masker, [('output_image', 'mask_file')]),
        (mnimask, bim, [('output_image', 'mask')]),
        (masker, bim, [('out_file', 'in_file')]),
        (bim, l1, [('out_file', 'functional_runs')]),
        (ev, l1, [('event_files', 'event_files')]),
        (conf2movpar, l1, [('out', 'realignment_parameters')]),
        (l1, l1model, [('session_info', 'session_info')]),
        (ev, l1model, [('orthogonalization', 'orthogonalization')]),
        (l1model, l1featmodel, [
            ('fsf_files', 'fsf_file'),
            ('ev_files', 'ev_files')]),
        (l1model, l1estimate, [('fsf_files', 'fsf_file')]),
        # Smooth
        (inputnode, pre_smooth, [('bold_preproc', 'zstat_file'),
                                 ('brainmask', 'mask_file')]),
        (bim, post_smooth, [('out_file', 'zstat_file')]),
        (inputnode, post_smooth, [('brainmask', 'mask_file')]),
        (pre_smooth, outputnode, [(('resels', _resels), 'sigma_pre')]),
        (post_smooth, outputnode, [(('resels', _resels), 'sigma_post')]),
    ])

    # Writing outputs
    csv = pe.Node(AddCSVRow(in_file=str(output_dir / 'smoothness.csv')),
                  name='addcsv_%s_%s' % (subject_id, pipeline))
    csv.inputs.sub_id = subject_id
    csv.inputs.pipeline = pipeline

    # Datasinks
    out_subject = Path(output_dir / 'sub-{}'.format(subject_id) / 'func')
    out_subject.mkdir(parents=True, exist_ok=True)
    ds_zstat = pe.Node(niu.Function(function=_feat_file),
                       name='ds_zstat')
    ds_zstat.inputs.path = 'stats/zstat11.nii.gz'
    ds_zstat.inputs.dest = out_subject / 'sub-{}_task-{}_variant-{}_zstat11.nii.gz'.format(
        subject_id, task_id, pipeline)

    ds_tstat = pe.Node(niu.Function(function=_feat_file),
                       name='ds_tstat')
    ds_tstat.inputs.path = 'stats/tstat11.nii.gz'
    ds_tstat.inputs.dest = out_subject / 'sub-{}_task-{}_variant-{}_tstat11.nii.gz'.format(
        subject_id, task_id, pipeline)

    workflow.connect([
        (outputnode, csv, [('sigma_pre', 'smooth_pre'),
                           ('sigma_post', 'smooth_post')]),
        (l1estimate, ds_zstat, [('feat_dir', 'feat_dir')]),
        (l1estimate, ds_tstat, [('feat_dir', 'feat_dir')]),
        (ds_zstat, outputnode, [('out', 'zstat')]),
    ])
    return workflow


def _feat_file(feat_dir, path, dest):
    from pathlib import Path
    from shutil import copy
    path = Path(feat_dir) / path
    if path.is_file():
        copy(str(path), str(dest))
        return str(dest)

    raise RuntimeError('feat file not found')


def _confounds2movpar(in_confounds):
    from os.path import abspath
    import numpy as np
    import pandas as pd
    dataframe = pd.read_csv(
        in_confounds,
        sep='\t',
        usecols=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']).fillna(value=0)

    out_name = abspath('motion.par')
    np.savetxt(out_name, dataframe.values, '%5.3f')
    return out_name


def create_contrasts(task):
    """
    Create a contrasts list
    """

    contrasts = []
    contrasts += [('Go', 'T', ['GO'], [1])]
    contrasts += [('GoRT', 'T', ['GO_rt'], [1])]
    contrasts += [('StopSuccess', 'T', ['STOP_SUCCESS'], [1])]
    contrasts += [('StopUnsuccess', 'T', ['STOP_UNSUCCESS'], [1])]
    contrasts += [('StopUnsuccessRT', 'T', ['STOP_UNSUCCESS_rt'], [1])]
    contrasts += [('Go-StopSuccess', 'T', ['GO', 'STOP_SUCCESS'], [1, -1])]
    contrasts += [('Go-StopUnsuccess', 'T', ['GO', 'STOP_UNSUCCESS'], [1, -1])]
    contrasts += [('StopSuccess-StopUnsuccess', 'T',
                   ['STOP_SUCCESS', 'STOP_UNSUCCESS'], [1, -1])]

    # add negative
    repl_w_neg = []
    for con in contrasts:
        if '-' not in con[0]:
            newname = 'neg_%s' % con[0]
        else:
            newname = "-".join(con[0].split("-")[::-1])
        new = (newname, 'T', con[2], [-x for x in con[3]])
        repl_w_neg.append(con)
        repl_w_neg.append(new)

    return repl_w_neg


def main():
    """Entry point"""
    opts = get_parser().parse_args()

    # Data dir
    bids_deriv_dir = Path(opts.bids_deriv_dir)
    bids_dir = Path(opts.bids_dir)

    # Create work directory
    work_dir = Path(opts.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory
    output_dir = Path(opts.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a subjects_list (accepts glob wildcards)
    subjects_list = []
    participant_label = [pl[4:] if pl.startswith('sub-') else pl
                         for pl in opts.participant_label]
    for part_label in participant_label:
        # Calculate subjects available from either pipeline
        fslsubj = [v.parent.name[4:]
                   for v in (bids_deriv_dir / 'fslfeat').glob(
                       'sub-{}/func'.format(part_label))]

        fprsubj = [v.parent.name[4:]
                   for v in (bids_deriv_dir / 'fmriprep').glob(
                       'sub-{}/func'.format(part_label))]

        subjects_list += list(set(fslsubj).intersection(fprsubj))

    subjects_list = list(set(subjects_list) - CNP_SUBJECT_BLACKLIST)

    if not subjects_list:
        raise RuntimeError('No subjects selected')

    tasks = [t[5:] if t.startswith('task-') else t
             for t in opts.task]

    if not tasks:
        raise NotImplementedError
        tasks = ['*']

    # Build up big workflow
    wf = pe.Workflow(name='level1')
    wf.base_dir = str(work_dir)

    for task_id, pipeline in product(tasks, ['fslfeat', 'fmriprep']):
        inputnode = pe.Node(niu.IdentityInterface(
            fields=['contrasts']),
            name='_'.join(('inputnode', pipeline, task_id)))
        inputnode.inputs.contrasts = create_contrasts(task_id)

        merge = pe.Node(niu.Merge(len(subjects_list)),
                        name='_'.join(('merge', pipeline, task_id)))

        for i, sub_id in enumerate(subjects_list):
            # build new workflow
            subwf = first_level_wf(pipeline, sub_id, task_id, output_dir)
            ftpl = str(
                bids_deriv_dir / pipeline / 'sub-{}'.format(sub_id) / 'func' /
                'sub-{}_task-{}_%s'.format(sub_id, task_id))
            subwf.inputs.inputnode.bold_preproc = ftpl % \
                'bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
            subwf.inputs.inputnode.confounds = ftpl % 'bold_confounds.tsv'
            subwf.inputs.inputnode.brainmask = ftpl % \
                'bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'
            subwf.inputs.inputnode.events_file = str(
                bids_dir / 'sub-{}'.format(sub_id) / 'func' /
                'sub-{}_task-{}_events.tsv'.format(sub_id, task_id))

            wf.connect([
                (inputnode, subwf, [
                    ('contrasts', 'inputnode.contrasts')]),
                (subwf, merge, [
                    ('outputnode.zstat', 'in%d' % (i + 1))]),
            ])

        acm = pe.Node(ACM(), name='_'.join(('acm', pipeline, task_id)))
        ds = pe.Node(nio.DataSink(
            base_directory=str(output_dir)),
            name='_'.join(('ds', 'acm', pipeline, task_id)))
        wf.connect([
            (merge, acm, [('out', 'in_files')]),
            (acm, ds, [('out_file', 'acm.@%s_%s' % (pipeline, task_id))]),
        ])

    print('Workflow built, start running ...')
    wf.run('MultiProc')
    return 0


if __name__ == '__main__':
    code = main()
    sys.exit(code)
