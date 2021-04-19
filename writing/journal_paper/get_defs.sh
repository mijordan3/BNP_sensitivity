#!/bin/bash

echo
echo Sections:
grep seclabel *.tex

echo
echo Equations:
grep eqlabel *.tex

echo
echo Definitions:
grep deflabel *.tex

echo
echo Assumptions:
grep assulabel *.tex

echo
echo Examples:
grep exlabel *.tex

echo
echo Lemmas:
grep lemlabel *.tex

echo
echo Corollaries:
grep corlabel *.tex

echo
echo Theorems:
grep thmlabel *.tex
