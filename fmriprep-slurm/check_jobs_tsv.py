#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
import subprocess as sp
import nibabel as nb
import numpy as np
import pandas as pd


def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description='Job array checker',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('tasks_list', action='store',
                        help='file listing the job-array tasks')
    parser.add_argument('job_id', action='store',
                        help='the job array id')

    return parser


def main():
    """Entry point"""
    opts = get_parser().parse_args()

    with open(opts.tasks_list) as tfh:
        data = tfh.readlines()

    part_anchor = data[0].split(' ').index('participant')
    part_start = data[0].split(' ').index('--participant_label') + 1
    job_id = opts.job_id

    jobs = []
    for i, line in enumerate(data):
        line_sp = line.split(' ')
        dataset = line_sp[part_anchor - 2].strip().split('/')[-1]
        part_end = part_start
        for j, arg in enumerate(line_sp[part_start:]):
            if arg.startswith('--'):
                part_end += j
                break

        participant = line_sp[part_start:part_end]
        jobs.append([dataset, participant, i + 1])

    fields = ['ds', 'participant', 'runtime', 'status', 'maxvm',
              'maxrss', 'n_bold', 'total_tr', 'avg_tr']
    outfile = open('%s.tsv' % job_id, 'w')
    print('\t'.join(fields), file=outfile)
    outfile.flush()
    tpl = ('\t'.join(['{%s}' % s for s in fields])).format
    for job in sorted(jobs):
        results = {
            'ds': job[0],
            'participant': job[1],
            'runtime': 'n/a',
            'status': 'n/a',
            'maxvm': 'n/a',
            'maxrss': 'n/a',
            'n_bold': 'n/a',
            'total_tr': 'n/a',
            'avg_tr': 'n/a',
        }

        # Check job status
        p = sp.run('sacct -o State,Elapsed,MaxVMSize,MaxRSS --noheader -j'.split(' ') + [
            '%s_%d' % (job_id, job[2])], stdout=sp.PIPE)
        stdout = p.stdout.decode().split('\n')
        if not stdout[0]:
            status = 'PENDING'
            runtime = '00:00:00'
        else:
            out = re.split(r'\s+', stdout[-3].strip())
            # Check status
            status = out[0]
            runtime = out[1]

            if status == 'CANCELLED':
                out = re.split(r'\s+', stdout[-4].strip())
                status += ' (%s)' % out[0]

        results['status'] = status
        results['runtime'] = runtime

        if status == 'PENDING':
            print(tpl(**results), file=outfile)
            outfile.flush()
            continue

        results['maxvm'] = 0.0
        results['maxrss'] = 0.0

        with open('%s-%d.out' % (job_id, job[2]), 'r') as olfh:
            outlog = olfh.readlines()

        files = []
        for line in outlog:
            line = line.strip().strip('\n')
            if line.startswith('Creating bold processing workflow for '):
                files.append(line[len(
                    'Creating bold processing workflow for '):].replace('"', '').split()[0])

        if files:
            trs = [nb.load(f).shape[3] for f in files]
            results['n_bold'] = len(files)
            results['total_tr'] = int(np.sum(trs))
            results['avg_tr'] = float(np.average(trs))

        print(tpl(**results), file=outfile)
        outfile.flush()

    outfile.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
