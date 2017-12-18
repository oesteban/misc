#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from glob import glob
import numpy as np
import json
import datalad

files = glob('sub-*/func/*_bold.json')


def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description='Fix SliceTiming',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('file_list', action='store', nargs='+',
                        help='list of files to be updated')
    parser.add_argument('--rm', action='store_true', default=False,
                        help='remove SliceTiming field')
    return parser


def main():
    """Entry point"""
    opts = get_parser().parse_args()
    for f in opts.file_list:
        with open(f) as fh:
            sidecar = json.load(fh)

        st = sidecar.pop('SliceTiming', None)
        if st is not None and not opts.rm:
            st = np.array(st, dtype=float)
            if not np.any(st < 0.0):
                sidecar["SliceTiming"] = (st * 1.e-3).tolist()

        # As per Yarik, it is better to just remove the file before writing.
        os.remove(f)
        with open(f, 'w') as fh:
            json.dump(sidecar, fh, indent=4)
        datalad.utils.rotree(f, ro=True, chmod_files=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
