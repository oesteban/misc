#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
First-level analysis of the CNP dataset
"""
import os
import sys
import shutil
from pathlib import Path
import pandas as pd
from warnings import warn
from nipype.interfaces.fsl import FEATModel, FEAT, Level1Design, maths, ApplyWarp
from nipype.pipeline.engine import Workflow, Node
from nipype.algorithms.modelgen import SpecifyModel
from utils import utils, get_config
from nipype.interfaces import afni


# def _nilearnmask(in_file, mask_file):
#     import os
#     import numpy as np
#     import nibabel as nb
#     from nipype.utils.filemanip import fname_presuffix
#     from nilearn.image import resample_to_img
#     out_file = fname_presuffix(in_file, '_brain', newpath=os.getcwd())
#     out_mask = fname_presuffix(in_file, '_brainmask', newpath=os.getcwd())
#     newmask = resample_to_img(mask_file, in_file,
#                               interpolation='nearest')
#     newmask.to_filename(out_mask)
#     nii = nb.load(in_file)
#     data = nii.get_data() * newmask.get_data()[..., np.newaxis]
#     nii = nii.__class__(data, nii.affine, nii.header).to_filename(out_file)
#     return out_file, out_mask

def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description='First level analysis',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('bids_dir', action='store', help='BIDS-Derivatives root directory')
    parser.add_argument('output_dir', action='store', help='derivatives folder')
    parser.add_argument('participant_level', action='store',
                        choices=['participant', 'group'], help='level of analysis')
    parser.add_argument('--participant-label', action='store', type=str,
                        help='subject id')
    parser.add_argument('-w', '--work-dir', action='store',
                        default=os.path.join(os.getcwd(), 'work'))
    parser.add_argument('--task', action='store', nargs='+', help='select task')
    return parser


def main():
    """Entry point"""
    opts = get_parser().parse_args()

    # Data dir
    bids_dir = Path(opts.bids_dir)

    # Create work directory
    work_dir = Path(opts.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory
    output_dir = Path(opts.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_label = opts.participant_label
    if participant_label and participant_label.startswith('sub-'):
        participant_label = participant_label[4:]

    if not participant_label:
        participant_label = '*'

    for task_id in opts.task:
        if task_id.startswith('task-'):
            task_id = task_id[5:]

        task_fname = 'sub-{}_task-{}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'.format(
            participant_label, task_id)
        squery = task_fname.split('_')[0]

        subjects_list = {}
        for pipeline in ['fslfeat', 'fmriprep']:
            pipeline_prep = (bids_dir / pipeline / squery / 'func' /
                             task_fname).glob()
            subjects_list[pipeline] = [str(f.name).split('_')[0] for f in pipeline_prep]

        subjects_list = list(set(subjects_list['fslfeat']).intersection(
            set(subjects_list['fmriprep'])))

        print(subjects_list)
        return 0


if __name__ == '__main__':
    code = main()
    sys.exit(code)



# # INPUT FILES AND FOLDERS
# BIDSDIR = args.bids_dir
# if BIDSDIR is None:
#     raise RuntimeError(
#         'No BIDS root directory was specified. Please provide it with '
#         'the --bids-dir argument or using the BIDSDIR env variable.')

# SUBJECT = args.subject

# cf = get_config.get_folders(args.prep_pipeline)

# for task_id in ['stopsignal']:
#     cf_files = get_config.get_files(args.prep_pipeline, SUBJECT, task_id)

#     bidssub = os.listdir(os.path.join(BIDSDIR, SUBJECT, 'func'))
#     taskfiles = [x for x in bidssub if task_id in x]
#     if len(taskfiles) == 0:  # if no files for this task are present: skip task
#         warn('No task "%s" found for subject "%s". Skipping.' % (task_id, SUBJECT))
#         continue

#     if not utils.check_exceptions(SUBJECT, task_id):
#         warn('Skipping subject "%s", task "%s".' % (SUBJECT, task_id))
#         continue

#     # CREATE OUTPUT DIRECTORIES
#     if not os.path.exists(cf['resdir']):
#         os.mkdir(cf['resdir'])

#     subdir = os.path.join(cf['resdir'], SUBJECT)
#     if not os.path.exists(subdir):
#         os.mkdir(subdir)

#     if os.path.exists(os.path.join(subdir, '%s.feat' % task_id)):
#         warn('Folder "%s" exists, skipping.' %
#              os.path.join(subdir, '%s.feat' % task_id))
#         continue

#     taskdir = os.path.join(cf['resdir'], SUBJECT, task_id)
#     if not os.path.exists(taskdir):
#         os.mkdir(taskdir)

#     eventsdir = os.path.join(taskdir, 'events')
#     if not os.path.exists(eventsdir):
#         os.mkdir(eventsdir)

#     os.chdir(taskdir)

#     # GENERATE task_id REGRESSORS, CONTRASTS + CONFOUNDERS
#     if args.prep_pipeline.startswith('fmriprep'):
#         confounds_infile = cf_files['confoundsfile']
#         confounds_in = pd.read_csv(confounds_infile, sep="\t")
#         confounds_in = confounds_in[['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']]
#         confoundsfile = utils.create_confounds(confounds_in, eventsdir)
#     else:
#         confoundsfile = cf_files['confoundsfile']

#     eventsfile = os.path.join(BIDSDIR, SUBJECT, 'func',
#                               '%s_task-%s_events.tsv' % (SUBJECT, task_id))

#     regressors = utils.create_ev_task(eventsfile, eventsdir, task_id)
#     EVfiles = regressors['EVfiles']
#     orthogonality = regressors['orthogonal']

#     contrasts = utils.create_contrasts(task_id)

#     # START PIPELINE
#     # inputmask = Node(IdentityInterface(fields=['mask_file']), name='inputmask')

#     if args.prep_pipeline.startswith("fsl"):
#         masker = Node(ApplyWarp(
#             in_file=cf_files['bold'],
#             field_file=cf_files['warpfile'],
#             ref_file=cf_files['standard'],
#             out_file=cf_files['masked'],
#             mask_file=cf_files['standard_mask']
#         ), name='masker')
#         # inputmask.inputs.mask_file = cf_files['standard_mask']
#     else:
#         masker = Node(maths.ApplyMask(
#             in_file=cf_files['bold'],
#             out_file=cf_files['masked'],
#             mask_file=cf_files['standard_mask']
#         ), name='masker')

#     bim = Node(afni.BlurInMask(
#         out_file=cf_files['smoothed'],
#         mask=cf_files['standard_mask'],
#         fwhm=5.0
#     ), name='bim')

#     l1 = Node(SpecifyModel(
#         event_files=EVfiles,
#         realignment_parameters=confoundsfile,
#         input_units='secs',
#         time_repetition=2,
#         high_pass_filter_cutoff=100
#     ), name='l1')

#     l1model = Node(Level1Design(
#         interscan_interval=2,
#         bases={'dgamma': {'derivs': True}},
#         model_serial_correlations=True,
#         orthogonalization=orthogonality,
#         contrasts=contrasts
#     ), name='l1design')

#     l1featmodel = Node(FEATModel(), name='l1model')

#     l1estimate = Node(FEAT(), name='l1estimate')

#     CNPflow = Workflow(name='cnp')
#     CNPflow.base_dir = taskdir
#     CNPflow.connect([
#         (masker, bim, [('out_file', 'in_file')]),
#         (bim, l1, [('out_file', 'functional_runs')]),
#         (l1, l1model, [('session_info', 'session_info')]),
#         (l1model, l1featmodel, [
#             ('fsf_files', 'fsf_file'),
#             ('ev_files', 'ev_files')]),
#         (l1model, l1estimate, [('fsf_files', 'fsf_file')])
#     ])

#     CNPflow.write_graph(graph2use='colored')
#     CNPflow.run('MultiProc', plugin_args={'n_procs': 4})

#     featdir = os.path.join(taskdir, "cnp", 'l1estimate', 'run0.feat')
#     utils.purge_feat(featdir)

#     shutil.move(featdir, os.path.join(subdir, '%s.feat' % task_id))
#     shutil.rmtree(taskdir)
