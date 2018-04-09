#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import nibabel as nb
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from nipype import logging

from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, SimpleInterface,
    traits, File, OutputMultiPath, isdefined,
    CommandLine, CommandLineInputSpec
)
from nipype.interfaces.ants.resampling import ApplyTransforms

logger = logging.getLogger('workflow')


class FixHeaderApplyTransforms(ApplyTransforms):
    """
    A replacement for nipype.interfaces.ants.resampling.ApplyTransforms that
    fixes the resampled image header to match the xform of the reference
    image
    """

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        # Run normally
        runtime = super(FixHeaderApplyTransforms, self)._run_interface(
            runtime, correct_return_codes)

        _copyxform(self.inputs.reference_image,
                   os.path.abspath(self._gen_filename('output_image')))
        return runtime


class FDRInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='-i %s',
                   desc='input pstat file')
    in_mask = File(exists=True, argstr='-m %s', desc='mask file')
    q_value = traits.Float(0.05, argstr='-q %f', usedefault=True,
                           desc='q-value (FDR) threshold')


class FDROutputSpec(TraitedSpec):
    fdr_val = traits.Float()


class FDR(CommandLine):
    _cmd = 'fdr'
    input_spec = FDRInputSpec
    output_spec = FDROutputSpec

    def _run_interface(self, runtime):
        self.terminal_output = 'file_split'
        runtime = super(FDR, self)._run_interface(runtime)
        fdr = float(runtime.stdout.splitlines()[1])
        setattr(self, 'result', fdr)
        return runtime

    def _list_outputs(self):
        return {'fdr_val': getattr(self, 'result')}


class PtoZInputSpec(CommandLineInputSpec):
    p_value = traits.Float(0.05, argstr='%f', usedefault=True, position=1,
                           desc='p-value (PtoZ) threshold')
    twotail = traits.Bool(False, argstr='-2', usedefault=True, position=2,
                          desc='use 2-tailed conversion (default is 1-tailed)')
    resels = traits.Float(argstr='-g %f', position=-1,
                          desc='use GRF maximum-height theory instead of Gaussian pdf')


class PtoZOutputSpec(TraitedSpec):
    z_val = traits.Float()


class PtoZ(CommandLine):
    _cmd = 'ptoz'
    input_spec = PtoZInputSpec
    output_spec = PtoZOutputSpec

    def _run_interface(self, runtime):
        self.terminal_output = 'file_split'
        runtime = super(PtoZ, self)._run_interface(runtime)
        zval = float(runtime.stdout.splitlines()[0])
        setattr(self, 'result', zval)
        return runtime

    def _list_outputs(self):
        return {'z_val': getattr(self, 'result')}


class CorrelationInputSpec(BaseInterfaceInputSpec):
    in_file1 = File(exists=True, mandatory=True, desc='input file 1')
    in_file2 = File(exists=True, mandatory=True, desc='input file 2')
    in_mask = File(exists=True, desc='input mask')
    metric = traits.Enum('pearson', 'distance', usedefault=True,
                         desc='correlation metric')
    subsample = traits.Float(100.0, usedefault=True)


class CorrelationOutputSpec(TraitedSpec):
    out_corr = traits.Float()


class Correlation(SimpleInterface):
    """
    """
    input_spec = CorrelationInputSpec
    output_spec = CorrelationOutputSpec

    def _run_interface(self, runtime):
        im1 = nb.load(self.inputs.in_file1).get_data()
        im2 = nb.load(self.inputs.in_file2).get_data()

        mask = np.ones_like(im1, dtype=bool)
        if isdefined(self.inputs.in_mask):
            mask = nb.load(
                self.inputs.in_mask).get_data() > 0.0

        if self.inputs.metric == 'pearson':
            corr = float(pearsonr(im1[mask], im2[mask])[0])
        else:
            if 0 < self.inputs.subsample < 100:
                nvox = int(mask.sum())
                logger.info('before: %d', nvox)
                size = int(nvox * self.inputs.subsample) // 100
                reshaped = np.zeros_like(mask[mask], dtype=bool)
                indexes = np.random.choice(
                    range(nvox), size=size, replace=False)
                reshaped[indexes] = True
                mask[mask] = reshaped
                logger.info('after: %d', mask.sum())
            corr = float(distcorr(im1[mask], im2[mask]))

        self._results['out_corr'] = corr
        return runtime


class EventsFilesForTaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, desc='input file, part of a BIDS tree')
    task = traits.Str(mandatory=True, desc='task')


class EventsFilesForTaskOutputSpec(TraitedSpec):
    event_files = OutputMultiPath(File(exists=True), desc='event files')
    orthogonalization = traits.Dict(int, traits.Dict(int, int),
                                    desc='orthogonalization')


class EventsFilesForTask(SimpleInterface):
    """
    """
    input_spec = EventsFilesForTaskInputSpec
    output_spec = EventsFilesForTaskOutputSpec

    def _run_interface(self, runtime):

        if self.inputs.task != 'stopsignal':
            raise NotImplementedError(
                'This function was not designed for tasks other than "stopsignal". '
                'Task "%s" cannot be processed' % self.inputs.task)

        events = pd.read_csv(self.inputs.in_file, sep="\t", na_values='n/a')
        self._results['event_files'] = []

        nEV = 6
        self._results['orthogonalization'] = {
            x: {y: 0 for y in range(1, nEV + 1)} for x in range(1, nEV + 1)
        }

        go_table = events[(events.TrialOutcome == "SuccessfulGo")]
        self._results['event_files'].append(
            create_ev(go_table, out_name="GO", duration=1, amplitude=1,
                      out_dir=runtime.cwd))
        self._results['event_files'].append(create_ev(
            go_table, out_name="GO_rt", duration='ReactionTime',
            amplitude=1, out_dir=runtime.cwd))
        self._results['orthogonalization'][2][1] = 1
        self._results['orthogonalization'][2][0] = 1

        stop_success_table = events[(events.TrialOutcome == "SuccessfulStop")]
        self._results['event_files'].append(create_ev(
            stop_success_table, out_name="STOP_SUCCESS",
            duration=1, amplitude=1, out_dir=runtime.cwd))

        stop_unsuccess_table = events[(events.TrialOutcome == "UnsuccessfulStop")]
        self._results['event_files'].append(create_ev(
            stop_unsuccess_table, out_name="STOP_UNSUCCESS",
            duration=1, amplitude=1, out_dir=runtime.cwd))
        self._results['event_files'].append(create_ev(
            stop_unsuccess_table, out_name="STOP_UNSUCCESS_rt",
            duration='ReactionTime', amplitude=1, out_dir=runtime.cwd))
        self._results['orthogonalization'][5][4] = 1
        self._results['orthogonalization'][5][0] = 1

        junk_table = events[(events.TrialOutcome == "JUNK")]
        if len(junk_table) > 0:
            self._results['event_files'].append(create_ev(
                junk_table, out_name="JUNK",
                duration=1, amplitude=1, out_dir=runtime.cwd))

        return runtime


def create_ev(dataframe, out_dir, out_name, duration=1, amplitude=1):
    """
    Adapt a BIDS-compliant events file to a format compatible with FSL feat
    Args:
        dataframe: events file from BIDS spec
        out_dir: path where new events file will be stored
        out_name: filename for the new events file
        amplitude: value or variable name
        duration: value or variable name
    Returns:
        Full path to the new events file
    """
    dataframe = dataframe[dataframe.onset.notnull()]
    dataframe.onset = dataframe.onset.round(3)

    if isinstance(duration, (float, int)):
        dataframe['duration'] = [duration] * len(dataframe)
    elif isinstance(duration, str):
        dataframe.duration = dataframe[[duration]].round(3)

    if isinstance(amplitude, (float, int)):
        dataframe['weights'] = [amplitude] * len(dataframe)
    elif isinstance(amplitude, str):
        dataframe['weights'] = dataframe[[amplitude]] - dataframe[[amplitude]].mean()
        dataframe.weights = dataframe.weights.round(3)

    # Prepare file
    ev_file = os.path.join(out_dir, '%s.txt' % out_name)
    dataframe[['onset', 'duration', 'weights']].to_csv(
        ev_file, sep="\t", header=False, index=False)
    return ev_file


def _copyxform(ref_image, out_image, message=None):
    # Read in reference and output
    resampled = nb.load(out_image)
    orig = nb.load(ref_image)

    # Copy xform infos
    qform, qform_code = orig.header.get_qform(coded=True)
    sform, sform_code = orig.header.get_sform(coded=True)
    header = resampled.header.copy()
    header.set_qform(qform, int(qform_code))
    header.set_sform(sform, int(sform_code))
    header['descrip'] = 'xform matrices modified by %s.' % (message or '(unknown)')

    newimg = resampled.__class__(resampled.get_data(), orig.affine, header)
    newimg.to_filename(out_image)


def distcorr(X, Y):
    """ Compute the distance correlation function

    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X).astype(float)
    Y = np.atleast_1d(Y).astype(float)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor
