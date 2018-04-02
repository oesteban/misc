#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import afni
from nipype.interfaces import fsl
from nipype.algorithms.modelgen import SpecifyModel
from nipype.algorithms.misc import AddCSVRow
from .interfaces import EventsFilesForTask


def first_level_wf(pipeline, subject_id, task_id, output_dir):
    """
    First level workflow
    """
    workflow = pe.Workflow(name='_'.join((pipeline, subject_id, task_id)))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_preproc', 'contrasts', 'confounds', 'brainmask', 'events_file']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['sigma_pre', 'sigma_post', 'out_stats']), name='outputnode')

    conf2movpar = pe.Node(niu.Function(function=_confounds2movpar),
                          name='conf2movpar')
    masker = pe.Node(fsl.ApplyMask(), name='masker')
    bim = pe.Node(afni.BlurInMask(fwhm=5.0, outputtype='NIFTI_GZ'),
                  name='bim', mem_gb=20)

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
    l1estimate = pe.Node(fsl.FEAT(), name='l1estimate', mem_gb=40)

    # pre_smooth = pe.Node(afni.FWHMx(combine=True, detrend=True),
    #                      name='smooth_pre')
    # post_smooth = pe.Node(afni.FWHMx(combine=True, detrend=True),
    #                       name='smooth_post')

    pre_smooth = pe.Node(fsl.SmoothEstimate(),
                         name='smooth_pre', mem_gb=20)
    post_smooth = pe.Node(fsl.SmoothEstimate(),
                          name='smooth_post', mem_gb=20)

    def _resels(val):
        return val ** (1 / 3.)

    # def _fwhm(fwhm):
    #     from numpy import mean
    #     return float(mean(fwhm, dtype=float))

    workflow.connect([
        (inputnode, masker, [('bold_preproc', 'in_file'),
                             ('brainmask', 'mask_file')]),
        (inputnode, ev, [('events_file', 'in_file')]),
        (inputnode, l1model, [('contrasts', 'contrasts')]),
        (inputnode, conf2movpar, [('confounds', 'in_confounds')]),
        (inputnode, bim, [('brainmask', 'mask')]),
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

        # Smooth with AFNI
        # (inputnode, pre_smooth, [('bold_preproc', 'in_file'),
        #                          ('brainmask', 'mask')]),
        # (bim, post_smooth, [('out_file', 'in_file')]),
        # (inputnode, post_smooth, [('brainmask', 'mask')]),
        # (pre_smooth, outputnode, [(('fwhm', _fwhm), 'sigma_pre')]),
        # (post_smooth, outputnode, [(('fwhm', _fwhm), 'sigma_post')]),
    ])

    # Writing outputs
    csv = pe.Node(AddCSVRow(in_file=str(output_dir / 'smoothness.csv')),
                  name='addcsv_%s_%s' % (subject_id, pipeline))
    csv.inputs.sub_id = subject_id
    csv.inputs.pipeline = pipeline

    # Datasinks
    ds_stats = pe.Node(niu.Function(function=_feat_stats),
                       name='ds_stats')
    ds_stats.inputs.subject_id = subject_id
    ds_stats.inputs.task_id = task_id
    ds_stats.inputs.variant = pipeline
    ds_stats.inputs.out_path = output_dir
    setattr(ds_stats.interface, '_always_run', True)

    workflow.connect([
        (outputnode, csv, [('sigma_pre', 'smooth_pre'),
                           ('sigma_post', 'smooth_post')]),
        (l1estimate, ds_stats, [('feat_dir', 'feat_dir')]),
        (ds_stats, outputnode, [('out', 'out_stats')]),
    ])
    return workflow


def _feat_stats(feat_dir, subject_id, task_id, variant, out_path):
    from pathlib import Path
    from shutil import copy

    out_names = []
    dest = out_path / 'sub-{}'.format(subject_id) / 'func' / \
        'sub-{}_task-{}_variant-{}_%s'.format(subject_id, task_id, variant)
    dest.parent.mkdir(parents=True, exist_ok=True)

    for statfile in (Path(feat_dir) / 'stats').glob('*.nii.gz'):
        out_names.append(str(dest) % statfile.name)
        copy(str(statfile), out_names[-1])
    return sorted(out_names)


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
