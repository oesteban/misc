#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
First-level analysis of the CNP dataset
"""
import os
import sys
from pathlib import Path
import warnings

# from warnings import warn
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from nipype.algorithms.stats import ACM
from nipype.algorithms.metrics import FuzzyOverlap

from .workflows import first_level_wf, second_level_wf
from .interfaces import Correlation
from nipype.algorithms.misc import AddCSVRow


warnings.simplefilter("once")

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
    parser.add_argument('--mem-gb', action='store', type=int, help='RAM in GB')
    parser.add_argument('--nprocs', action='store', type=int, help='number of processors')
    parser.add_argument('--repetition', action='store', type=int, default=1,
                        help='repetition number')
    return parser


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
    from multiprocessing import set_start_method
    set_start_method('forkserver')

    opts = get_parser().parse_args()

    plugin_args = {
        'raise_insufficient': False,
        'memory_gb': opts.mem_gb,
        'n_procs': opts.nprocs
    }
    plugin_args = {k: v for k, v in plugin_args.items()
                   if v is not None}

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
    if opts.participant_level == 'participant':
        wf = first_level(subjects_list, tasks, output_dir, bids_dir,
                         bids_deriv_dir)
    else:
        start = 10
        wf = pe.Workflow('level2')
        groupsizes = _arange(start, len(subjects_list) // 2, 5)
        for ss in groupsizes:
            swf = second_level(subjects_list, tasks, output_dir, 11,
                               bids_deriv_dir, sample_size=ss,
                               repetition=opts.repetition)
            wf.add_nodes([swf])

    wf.base_dir = str(work_dir)
    print('Workflow built, start running ...')
    wf.run('MultiProc', plugin_args=plugin_args)
    return 0


def _arange(start, end, increment):
    from numpy import arange
    return arange(start, end, increment).tolist()


def first_level(subjects_list, tasks_list, output_dir,
                bids_dir, bids_deriv_dir):
    wf = pe.Workflow(name='level1')
    for task_id in tasks_list:
        inputnode = pe.Node(niu.IdentityInterface(
            fields=['contrasts']),
            name='_'.join(('inputnode', task_id)))
        inputnode.inputs.contrasts = create_contrasts(task_id)

        # Run pipelines
        for pipeline in ['fslfeat', 'fmriprep']:
            nsubjects = len(subjects_list)
            if nsubjects > 1:
                merge = pe.Node(niu.Merge(nsubjects),
                                name='_'.join(('merge', pipeline, task_id)))
                acm = pe.Node(ACM(), name='_'.join(('acm', pipeline, task_id)))
                ds = pe.Node(nio.DataSink(
                    base_directory=str(output_dir)),
                    name='_'.join(('ds', 'acm', pipeline, task_id)))
                wf.connect([
                    (merge, acm, [('out', 'in_files')]),
                    (acm, ds, [('out_file', 'acm.@%s_%s' % (pipeline, task_id))]),
                ])

            for i, sub_id in enumerate(subjects_list):
                # build new workflow
                subwf = first_level_wf(pipeline, sub_id, task_id, output_dir)
                ftpl = str(
                    bids_deriv_dir / pipeline / 'sub-{}'.format(sub_id) / 'func' /
                    'sub-{}_task-{}_%s'.format(sub_id, task_id))
                subwf.inputs.inputnode.bold_preproc = ftpl % \
                    'bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
                subwf.inputs.inputnode.confounds = ftpl % 'bold_confounds.tsv'
                subwf.inputs.inputnode.events_file = str(
                    bids_dir / 'sub-{}'.format(sub_id) / 'func' /
                    'sub-{}_task-{}_events.tsv'.format(sub_id, task_id))
                subwf.inputs.inputnode.brainmask = ftpl % \
                    'bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'


                wf.connect([
                    (inputnode, subwf, [
                        ('contrasts', 'inputnode.contrasts')]),
                ])

                # # Connect brain mask
                # if pipeline == 'fslfeat':
                #     subwf.inputs.inputnode.brainmask = ftpl % \
                #         'bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'
                # else:
                #     subwf.inputs.inputnode.brainmask = str(
                #         bids_deriv_dir / 'fmriprep' / 'mni_resampled_brainmask.nii.gz'
                #     )

                # Connect merger
                if nsubjects > 1:
                    wf.connect([
                        (subwf, merge, [
                            ('outputnode.zstat', 'in%d' % (i + 1))]),
                    ])
    return wf


def second_level(subjects_list, tasks_list, output_dir, contrast_id,
                 bids_deriv_dir, sample_size=None, repetition=1):
    import numpy as np
    if not sample_size:
        raise RuntimeError

    if sample_size * 2 > len(subjects_list):
        raise RuntimeError('Sample size too big')

    np.random.seed(np.uint32(12345 * (repetition + 1) * sample_size))
    full_sample = np.random.choice(sorted(subjects_list), sample_size * 2,
                                   replace=False)
    groups = [sorted(full_sample[:sample_size].tolist()),
              sorted(full_sample[sample_size:].tolist())]

    group_output = output_dir.parent / 'l2'
    group_output.mkdir(parents=True, exist_ok=True)

    wf = pe.Workflow(name='level2_N%03d' % sample_size)
    for task_id in tasks_list:
        for pipeline in ['fmriprep', 'fslfeat']:
            fdice = pe.Node(FuzzyOverlap(weighting='volume'),
                            name='_'.join(('fuzzy_dice', task_id, pipeline)))
            bdice = pe.Node(FuzzyOverlap(weighting='volume'),
                            name='_'.join(('binary_dice', task_id, pipeline)))

            corr = pe.Node(Correlation(),
                           name='_'.join(('corr', task_id, pipeline)))
            # dcorr = pe.Node(Correlation(metric='distance', subsample=1.),
            #                 mem_gb=60, name='_'.join(('dcorr', task_id, pipeline)))

            tocsv = pe.Node(AddCSVRow(
                in_file=str(group_output / 'group.csv')),
                name='_'.join(('tcsv', task_id, pipeline)),
                mem_gb=64)
            tocsv.inputs.pipeline = pipeline
            tocsv.inputs.N = sample_size
            tocsv.inputs.repetition = repetition

            for gid, group in enumerate(groups):
                group_pattern = [
                    str(output_dir / 'sub-{}'.format(s) / 'func' /
                        'sub-{}_task-{}_variant-{}_%s{:d}.nii.gz'.format(
                            s, task_id, pipeline, contrast_id))
                    for s in group
                ]
                subwf = second_level_wf(
                    name='_'.join((pipeline, task_id, 'N%03d' % sample_size, 'S%d' % gid)))
                subwf.inputs.inputnode.copes = [pat % 'cope' for pat in group_pattern]
                subwf.inputs.inputnode.varcopes = [pat % 'varcope' for pat in group_pattern]

                ds = pe.Node(nio.DataSink(base_directory=str(group_output)),
                             name='_'.join(('ds', task_id, pipeline, 'N%d' % sample_size,
                                            'G%d' % gid)))
                dsfmt = ('%s_%s_N%03d_R%03d_S%d.@{}' % (
                         pipeline, task_id, sample_size, repetition, gid)).format
                group_mask = str(
                    Path.home() / '.cache' / 'stanford-crn' /
                    'mni_icbm152_nlin_asym_09c' / '2mm_brainmask.nii.gz')

                # # Connect brain mask
                # if pipeline == 'fslfeat':
                #     group_mask = str(
                #         Path.home() / '.cache' / 'stanford-crn' /
                #         'mni_icbm152_nlin_asym_09c' / '2mm_brainmask.nii.gz')
                # else:
                #     group_mask = str(
                #         bids_deriv_dir / 'fmriprep' / 'mni_resampled_brainmask.nii.gz'
                #     )
                subwf.inputs.inputnode.group_mask = group_mask
                corr.inputs.in_mask = group_mask
                wf.connect([
                    (subwf, fdice, [
                        (('outputnode.pstat', _tolist), 'in_ref' * gid or 'in_tst')]),
                    (subwf, bdice, [
                        (('outputnode.fdr_thres', _tolist), 'in_ref' * gid or 'in_tst')]),
                    (subwf, corr, [
                        ('outputnode.tstat', 'in_file%d' % (gid + 1))]),
                    # (subwf, dcorr, [
                    #     ('outputnode.tstat', 'in_file%d' % (gid + 1))]),
                    (subwf, ds, [
                        ('outputnode.pstat', dsfmt('pstat')),
                        ('outputnode.tstat', dsfmt('tstat')),
                        ('outputnode.zstat', dsfmt('zstat')),
                        ('outputnode.fdr_thres', dsfmt('fdr_thres')),
                        ('outputnode.fwe_thres', dsfmt('fwe_thres')),
                    ])
                ])

            wf.connect([
                (fdice, tocsv, [('dice', 'fdice')]),
                (bdice, tocsv, [('dice', 'bdice')]),
                (corr, tocsv, [('out_corr', 'correlation')]),
                # (dcorr, tocsv, [('out_corr', 'corrdist')]),
            ])

    return wf


def _tolist(inval):
    return [inval]


if __name__ == '__main__':
    code = main()
    sys.exit(code)
