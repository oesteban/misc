#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import nibabel as nb

from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, SimpleInterface,
    traits, File, OutputMultiPath
)
from nipype.interfaces.ants.resampling import ApplyTransforms


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
