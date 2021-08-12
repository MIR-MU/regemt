#!/bin/python

""" Reads a WMT21 Metrics task segment-level submission file, fixes common issues, and prints it back. """

import sys


for line in sys.stdin:
    metric, lp, testset, refset, sysid, segid, score = line.split('\t')
    segid = str(int(segid) + 1)
    if refset.startswith('ref-') and not sysid.startswith('ref-'):
        refset = 'src'
    if sysid.startswith('hyp.') or sysid.startswith('ref.'):
        sysid = sysid[4:]
    if sysid == 'Allegro.eu':
        sysid = 'Allegro'
    print('\t'.join([metric, lp, testset, refset, sysid, segid, score]), end='')
