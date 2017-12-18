#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys


SCRIPT_TEMPLATE = """
<script>
function download() {{
    var ds = "{dataset}"
    var sub = "{subject}"
    var reportlets = []

    var els = document.getElementsByClassName("rating-form")
    for (var elid = 0; elid < els.length; elid++) {{
        // Do stuff here
        var thisrep = {{}}
        for (var i = 0; i < els[elid].children.length; i++) {{
            thisrep[els[elid].children[i].name] = els[elid].children[i].value
        }}

        reportlets.push(thisrep)
    }};

    var file = new Blob([JSON.stringify({{
        'dataset': ds,
        'subject': sub,
        'reports': reportlets}}
    )], {{type: 'text/json'}});

    var a = document.getElementById("a");
    a.href = URL.createObjectURL(file);
    a.download = ds + "_" + sub + ".json";
    a.style.visibility = "visible"
}}
</script>
""".format

BUTTON_TEMPLATE = """
<button onclick="download()">Create file</button>
<a href="" id="a" style="visibility:hidden">Save your ratings</a>
<br />
<br />
"""

RATE_TEMPLATE = """
<div class="rating-form">
<input type="hidden" value="{reportlet}" name="name" />
<select name="rating">
  <option value="-1">(unrated)</option>
  <option value="0">Useless</option>
  <option value="1">Poor result</option>
  <option value="2">Acceptable</option>
  <option value="3">Exceptional</option>
</select>
<textarea rows="2" cols="100" name="comment">
</textarea>
</div>
""".format


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


def inject_js(in_file, out_file, dataset):
    """parse the html and inject code"""
    with open(in_file) as rfh:
        content = rfh.read()

    sub = os.path.splitext(
        os.path.basename(in_file))[0]

    out_lines = []
    infname = False
    lastfield = None
    nextinject = None
    finished = None
    for line in content.splitlines():
        if line.strip() == "</head>":
            out_lines += SCRIPT_TEMPLATE(
                dataset=dataset,
                subject=sub).splitlines()

        if infname and lastfield is None:
            lastfield = os.path.basename(line.strip()[10:-4])

        if infname and lastfield is not None:
            nextinject = RATE_TEMPLATE(
                reportlet=lastfield).splitlines()
            lastfield = None
            infname = None

        if line.strip() == '<div class="elem-filename">':
            infname = True

        if infname and line.strip() == '</div>':
            infname = False

        if line.strip() == '<div id="errors">':
            finished = False

        if finished is False and line.strip() == '</div>':
            finished = True

        out_lines.append(line)

        if nextinject is not None:
            out_lines += nextinject
            nextinject = None

        if finished:
            out_lines += ['<h1>Overall rating and comments</h1>']
            out_lines += RATE_TEMPLATE(
                reportlet="overall").splitlines()
            out_lines += BUTTON_TEMPLATE.splitlines()
            finished = None

    with open(out_file, 'w') as rfh:
        rfh.write('\n'.join(out_lines))


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
        derivs = os.path.expandvars(line_sp[part_anchor - 1].strip())
        dataset = line_sp[part_anchor - 2].strip().split('/')[-1]
        part_end = part_start
        for j, arg in enumerate(line_sp[part_start:]):
            if arg.startswith('--'):
                part_end += j
                break

        participant = line_sp[part_start:part_end]
        jobs.append([dataset, participant, derivs])

    for ds, sub, derivs in sorted(jobs):
        fname = os.path.join(derivs, 'fmriprep', sub[0])

        if os.path.isfile('%s.html' % fname):
            inject_js('%s.html' % fname, '%s-rater.html' % fname, ds)

    return 0


if __name__ == '__main__':
    sys.exit(main())
