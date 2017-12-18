#!/bin/bash

for i in $( find . -name *-rater.html ); do newdir="$SCRATCH/reports/$(dirname ${i:2:10})/$( basename ${i::${#i}-11} )"; mkdir -p $newdir; cp $i $SCRATCH/reports/$(dirname ${i:2:10})/$(basename $i); cp -r ${i::${#i}-11}/figures $newdir/; done
