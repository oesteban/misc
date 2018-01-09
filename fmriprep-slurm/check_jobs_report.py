#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
import datetime
import subprocess as sp
from textwrap import indent


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
    parser.add_argument('-M', '--missing-tasks-list', action='store',
                        help='write a new tasks_list.sh file with failed tasks')

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
        jobs.append([dataset, i + 1, participant, line.strip('\n')])

    rst_report = 'Job Array Report (job ID %s)' % job_id
    rst_report = '\n%s\n%s\n\n' % (rst_report, '=' * len(rst_report))

    last_ds = None
    for job in sorted(jobs):
        if last_ds != job[0]:
            rst_report += '\n\n%s\n%s\n\n' % (job[0], '-' * len(job[0]))
            last_ds = job[0]
        p = sp.run('sacct -o State,Elapsed,MaxVMSize,MaxRSS --noheader -j'.split(' ') + [
            '%s_%d' % (job_id, job[1])], stdout=sp.PIPE)
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

        if status != 'COMPLETED' and opts.missing_tasks_list:
            with open(opts.missing_tasks_list, 'a') as f:
                print(job[-1], file=f)

        subs = 'Subjects: %s - status %s - runtime %s' % (', '.join(job[2]), status, runtime)
        rst_report += '%s\n%s\n\n' % (subs, '~' * len(subs))

        if status == 'PENDING':
            continue

        with open('%s-%d.out' % (job_id, job[1]), 'r') as olfh:
            outlog = olfh.readlines()

        nipype_t1 = None
        nipype_t2 = None
        if outlog and outlog[0].strip().split(',')[0]:
            nipype_t1 = datetime.datetime.strptime(outlog[0].strip().split(
                ',')[0], '%y%m%d-%H:%M:%S')
        for line in reversed(outlog):
            datechunk = line.strip().split(',')[0]
            if datechunk:
                try:
                    nipype_t2 = datetime.datetime.strptime(datechunk, '%y%m%d-%H:%M:%S')
                    break
                except ValueError:
                    pass

        if nipype_t1 and nipype_t2:
            rst_report += '* Nipype log elapsed time: %s.\n' % (nipype_t2 - nipype_t1)
        else:
            rst_report += '* Nipype log missing or failed reading times.\n'

        files = []
        for line in outlog:
            line = line.strip().strip('\n')
            if line.startswith('Creating bold processing workflow for '):
                files.append(line[len(
                    'Creating bold processing workflow for '):].replace('"', '``'))
            if 'ERROR' in line:
                rst_report += '* ``ERROR`` found in log\n'
            if line.startswith('Saving crash'):
                cfname = line.split(' ')[-1]
                with open(cfname, 'r') as cffh:
                    crash = cffh.read()

                rst_report += '* Crashfile: ``%s`` ::\n\n' % cfname
                rst_report += indent(crash, ' ' * 4) + '\n\n'

        with open('%s-%d.err' % (job_id, job[1]), 'r') as elfh:
            errlog = [l.strip('\n').strip() for l in elfh.readlines() if l.strip('\n').strip()]

        if errlog:
            rst_report += '* Slurm error log: ::\n\n' + indent('\n'.join(errlog), ' ' * 4) + '\n\n'

        if files:
            rst_report += '* BOLD - %d files:\n\n%s\n\n' % (
                len(files), indent('\n'.join(sorted(files)), ' ' * 2 + '- '))

    # Save rst file
    with open('%s.rst' % job_id, 'w') as outfile:
        outfile.write(rst_report)

    # Convert to html
    sp.run(['rst2html.py', '%s.rst' % job_id, '%s.html' % job_id])

    return 0


if __name__ == '__main__':
    sys.exit(main())
