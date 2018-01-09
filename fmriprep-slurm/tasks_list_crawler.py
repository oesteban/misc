# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Tasks list crawler
"""
import os
import glob
import json

import numpy as np
import nibabel as nb


def parse_tasks_list(tasks_list_file, group=None):
    with open(tasks_list_file) as tfh:
        tasks = tfh.readlines()
    part_anchor = tasks[0].split(' ').index('participant')
    part_start = tasks[0].split(' ').index('--participant_label') + 1
    datasets = {}
    for line in tasks:
        line_sp = line.split(' ')
        ds = line_sp[part_anchor - 2].strip().split('/')[-1]
        if group is None:
            datasets.setdefault(ds, [])
        else:
            datasets.setdefault(ds, {group: []})

        part_end = part_start
        for j, arg in enumerate(line_sp[part_start:]):
            if arg.startswith('--'):
                part_end += j
                break

        part = line_sp[part_start:part_end]
        part = sorted([sub[4:] if sub.startswith('sub-') else sub for sub in part])
        if group is None:
            datasets[ds] += part
        else:
            datasets[ds][group] += part
    return datasets


def read_phases(tasks_list_phase1, tasks_list_phase2):
    ph1 = parse_tasks_list(tasks_list_phase1)
    ph2 = parse_tasks_list(tasks_list_phase2)

    newdict = {}
    for ds, subs in ph2.items():
        newdict.setdefault(ds, {'phase2': subs})
        if ds in ph1:
            newdict[ds]['phase1'] = ph1[ds]
    return newdict


def fill_metadata(alldict):
    npart = 0
    nruns = 0
    ntrs = 0
    for ds, ph in alldict.items():
        sessions = sum([os.path.isdir(item) for item in glob.glob(
            os.path.join(ds, 'sub-%s' % ph['phase2'][0], 'ses-*'))]) or 1
        alldict[ds]['sessions'] = sessions
        part = list(set(ph.get('phase1', []) + ph['phase2']))
        alldict[ds]['participants'] = part
        npart += len(part)

        metafiles = (sorted(glob.iglob(os.path.join(ds, '**', '*_T1w.json'))) +
                     sorted(glob.iglob(os.path.join(ds, '**', '*_bold.json'))) +
                     sorted(glob.glob(os.path.join(ds, '*.json'))))

        scanner = []
        for mf in metafiles:
            with open(mf) as mfh:
                try:
                    meta = json.load(mfh)
                except json.JSONDecodeError:
                    meta = {}
            scanner.append(meta.get('Manufacturer', None))

        boldmeta = (sorted(glob.glob(os.path.join(ds, '*_bold.json'))) +
                    sorted(glob.iglob(os.path.join(ds, '**', '*_bold.json'))))
        for bmf in boldmeta:
            trvals = []
            with open(bmf) as bmfh:
                meta = json.load(bmfh)
            trvals.append(meta.get('RepetitionTime'))
        trvals = list(set(['%.1f' % float(v) for v in trvals if v is not None]))

        if trvals:
            alldict[ds]['TR'] = trvals[0]

        alldict[ds]['scanner'] = ', '.join(
            set([s.upper() for s in scanner if s is not None])) or 'N/A'

        t1runs = []
        t2runs = []
        fmruns = []
        boldruns = []
        runs = []
        res = []
        dstrs = 0
        tasks = []
        for sub in part:
            bids = [ds, 'sub-%s' % sub, 'func', '*_bold.nii*']
            if sessions > 1:
                bids.insert(2, 'ses-*')

            struc = os.path.join(*bids)
            subruns = sorted(glob.glob(struc))
            runs.append(subruns)

            t1runs += sorted(glob.glob(os.path.join(*(bids[:-2] + ['anat', '*_T1w.nii*']))))
            t2runs += sorted(glob.glob(os.path.join(*(bids[:-2] + ['anat', '*_T2w.nii*']))))
            fmruns += (sorted(glob.glob(os.path.join(*(bids[:-2] + ['fmap', '*_epi.nii*']))) +
                       glob.glob(os.path.join(*(bids[:-2] + ['fmap', '*_phasediff.nii*']))) +
                       glob.glob(os.path.join(*(bids[:-2] + ['fmap', '*_phase1.nii*'])))))
            boldruns += subruns

            trs = []
            for r in subruns:
                nii = nb.load(r)
                trs.append(nii.shape[-1])
                res.append(nii.header.get_zooms()[:4])
                tasks += [v[len('task-'):] for v in os.path.basename(r).split('_')
                          if v.startswith('task-')]

            subtrs = sum(trs)
            ntrs += subtrs
            dstrs += subtrs

        alldict[ds]['tasks'] = len(set(tasks))
        alldict[ds]['modalities'] = {'T1w': int(len(t1runs) / len(part)),
                                     'T2w': int(len(t2runs) / len(part)),
                                     'FM': int(len(fmruns) / len(part)),
                                     'BOLD': int(len(boldruns) / len(part))}
        alldict[ds]['resolution'] = np.around(np.mean(res, axis=0)[:3], 2).tolist()
        alldict[ds]['num_trs'] = dstrs
        alldict[ds]['runs'] = sum([len(r) for r in runs])
        nruns += alldict[ds]['runs']

    return alldict


def dict2latex(alldict):
    rowformat = """\
    {ds} & {scanner} & {sessions} & {tasks} & {runs} &
           {mods} & {part_ph1} & {part_ph2} & {unique} &
           {tr} & {num_trs} & {resolution} \\\\\
    """.format

    for ds, vals in sorted(alldict.items()):
        line = rowformat(
            ds=ds.upper(),
            scanner=vals['scanner'],
            sessions=vals['sessions'],
            tasks=vals['tasks'],
            runs=vals['runs'],
            mods=', '.join(['%d~%s' % (v, k) for k, v in vals['modalities'].items() if v > 0]),
            part_ph1=', '.join(sorted(vals.get('phase1', []))),
            part_ph2=', '.join(sorted(vals['phase2'])),
            unique=len(vals['participants']),
            tr=vals.get('TR', 'N/A'),
            num_trs=vals['num_trs'],
            resolution='$\\times$'.join(['%.2f' % f for f in vals['resolution']])
        )

