Before knitting the knitr files, run in this directory

```
./R_scripts/sync_data.sh # pulls data from SCF /scratch/ folder
./R_scripts/process_data.sh # converts .npz files to .RData files
```

Then, run 
```
./knit_iris_to_tex.sh
./knit_mice_data_to_tex.sh
./knit_structure_to_tex.sh
```

to convert the `.rnw` files to `.tex` files for the Results section of the paper. 
