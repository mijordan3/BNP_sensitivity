#!/bin/bash
for FILE in introduction.tex abstract.tex; do
    aspell --per-conf=./aspell.conf --dont-backup -t -c $FILE;
done
