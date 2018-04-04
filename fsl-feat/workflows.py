#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import afni
from nipype.interfaces import fsl
from nipype.algorithms.modelgen import SpecifyModel
from nipype.algorithms.misc import AddCSVRow
from .interfaces import EventsFilesForTask, FDR, PtoZ


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


def second_level_wf(name='level2'):
    """second level analysis"""
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['copes', 'varcopes', 'group_mask']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['zstat', 'tstat', 'pstat', 'fwe_thres', 'fdr_thres']),
        name='outputnode')

    copemerge = pe.Node(fsl.Merge(dimension='t'), name='copemerge')
    varcopemerge = pe.Node(fsl.Merge(dimension='t'), name='varcopemerge')
    level2model = pe.Node(fsl.L2Model(), name='l2model')
    flameo = pe.Node(fsl.FLAMEO(run_mode='ols'), name='flameo')
    ztopval = pe.Node(fsl.ImageMaths(op_string='-ztop', suffix='_pval'),
                      name='ztop')

    # FDR
    fdr = pe.Node(FDR(), name='calc_fdr')
    fdr_apply = pe.Node(fsl.ImageMaths(
        suffix='_thresh_vox_fdr_pstat1'), name='fdr_apply')

    # FWE
    def _reselcount(voxels, resels):
        return float(voxels / resels)

    smoothness = pe.Node(fsl.SmoothEstimate(), name='smoothness')
    rescount = pe.Node(niu.Function(function=_reselcount), name='reselcount')
    ptoz = pe.Node(PtoZ(), name='ptoz')
    fwethres = pe.Node(fsl.Threshold(), name='fwethres')

    # Cluster
    cluster = pe.Node(fsl.Cluster(
        threshold=3.2, pthreshold=0.05, connectivity=26, use_mm=True),
        name='cluster')

    def _len(inlist):
        return len(inlist)

    def _lastidx(inlist):
        return len(inlist) - 1

    def _first(inlist):
        if isinstance(inlist, (list, tuple)):
            return inlist[0]
        return inlist

    def _fdr_thres_operator(fdr_th):
        return '-mul -1 -add 1 -thr %f' % (1 - fdr_th)

    # create workflow
    workflow.connect([
        (inputnode, copemerge, [('copes', 'in_files')]),
        (inputnode, varcopemerge, [('varcopes', 'in_files')]),
        (inputnode, level2model, [(('copes', _len), 'num_copes')]),
        (inputnode, flameo, [('group_mask', 'mask_file')]),
        (copemerge, flameo, [('merged_file', 'cope_file')]),
        (varcopemerge, flameo, [('merged_file', 'var_cope_file')]),
        (level2model, flameo, [
            ('design_mat', 'design_file'),
            ('design_con', 't_con_file'),
            ('design_grp', 'cov_split_file')]),
        (flameo, ztopval, [(('zstats', _first), 'in_file')]),
        (ztopval, fdr, [('out_file', 'in_file')]),
        (inputnode, fdr, [('group_mask', 'in_mask')]),
        (inputnode, fdr_apply, [('group_mask', 'mask_file')]),
        (flameo, fdr_apply, [(('zstats', _first), 'in_file')]),
        (fdr, fdr_apply, [
            (('fdr_val', _fdr_thres_operator), 'op_string')]),
        (inputnode, smoothness, [('group_mask', 'mask_file')]),
        (flameo, smoothness, [(('res4d', _first), 'residual_fit_file')]),
        (inputnode, smoothness, [(('copes', _lastidx), 'dof')]),
        (smoothness, rescount, [('resels', 'resels'),
                                ('volume', 'voxels')]),
        (rescount, ptoz, [('out', 'resels')]),
        (flameo, fwethres, [(('zstats', _first), 'in_file')]),
        (ptoz, fwethres, [('z_val', 'thresh')]),
        (flameo, cluster, [(('zstats', _first), 'in_file'),
                           (('copes', _first), 'cope_file')]),
        (smoothness, cluster, [('dlh', 'dlh'),
                               ('volume', 'volume')]),
        (flameo, outputnode, [
            (('zstats', _first), 'zstat'),
            (('tstats', _first), 'tstat'),
        ]),
        (ztopval, outputnode, [('out_file', 'pstat')]),
        (fdr_apply, outputnode, [('out_file', 'fdr_thres')]),
        (fwethres, outputnode, [('out_file', 'fwe_thres')]),
    ])
    return workflow


def _feat_stats(feat_dir, subject_id, task_id, variant, out_path):
    from pathlib import Path
    from shutil import copy

    out_names = []
    dest = out_path / 'sub-{}'.format(subject_id) / 'func' / \
        'sub-{}_task-{}_variant-{}_%s'.format(subject_id, task_id, variant)
    dest.parent.mkdir(parents=True, exist_ok=True)

    for statfile in Path(Path(feat_dir) / 'stats').glob('*.nii.gz'):
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
