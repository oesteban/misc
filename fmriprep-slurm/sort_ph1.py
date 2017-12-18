#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import re
import datetime
import subprocess as sp
from textwrap import indent

def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description='Organizer for FMRIPREP outputs',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('tasks_list', action='store',
                        help='file listing the job-array tasks')
    parser.add_argument('archive_dir', action='store',
                        help='where the outputs should be stored')

    return parser

def main():
    """Entry point"""
    opts = get_parser().parse_args()

    with open(opts.tasks_list) as tfh:
        data = tfh.readlines()
    
    part_anchor = data[0].split(' ').index('participant')
    part_start = data[0].split(' ').index('--participant_label') + 1
    jobs = []
    for i, line in enumerate(data):
        line_sp = line.split(' ')
        dataset = line_sp[part_anchor - 2].strip().split('/')[-1]
        output_dir = os.path.expandvars(line_sp[part_anchor - 1])
        part_end = part_start
        for j, arg in enumerate(line_sp[part_start:]):
            if arg.startswith('--'):
                part_end += j
                break
            
        participant = line_sp[part_start:part_end]
        for p in participant:
            os.symlink(os.path.join(output_dir, 'fmriprep', 'sub-%s.html' % p),
                       os.path.join(opts.archive_dir, '%s_sub-%s.html' % (dataset, p)))

    return 0

if __name__ == '__main__':
    sys.exit(main())
