#!/bin/bash
for FILE in `ls *.tex`; do
    aspell --per-conf=./aspell.conf --dont-backup -t -c $FILE;
done;
