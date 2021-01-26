#!/bin/bash

for FILE in `ls -1 *tex | grep -v ^_.*`; do
    echo Spell checking $FILE;
    aspell --per-conf=./aspell.conf --dont-backup -t -c $FILE;
done
