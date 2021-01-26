#!/bin/bash

echo
echo Sections:
grep seclabel appendices/*.tex *.tex

echo
echo Equations:
grep eqlabel appendices/*.tex *.tex

echo
echo Definitions:
grep deflabel appendices/*.tex *.tex

echo
echo Assumptions:
grep assulabel appendices/*.tex *.tex

echo
echo Conditions:
grep condlabel appendices/*.tex *.tex

echo
echo Propositions:
grep proplabel appendices/*.tex *.tex

echo
echo Lemmas:
grep lemlabel appendices/*.tex *.tex

echo
echo Corollaries:
grep corlabel appendices/*.tex *.tex

echo
echo Theorems:
grep thmlabel appendices/*.tex *.tex
