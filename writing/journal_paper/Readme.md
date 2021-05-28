The experiment files are all written using R knitr. They load in data from our experiments and create figures in code chunks in between text. 

To download the necessary data for the knitr files, run 
```
./R_scripts/sync_data.sh # pulls data from SCF /scratch/ folder
./R_scripts/process_data.sh # converts .npz files to .RData files
```
The second step saves the data into the `processed_data` folder as `.RData` files, which the knitr files subsequently load. 

**All writing for the experimental results is done in the .Rnw files**. The relevant knitr files for the experiments are: 
- `experiments_iris.Rnw`
- `experiments_mice.Rnw`
- `experiments_structure.Rnw` 
- and `timing_table.Rnw`


Then, to convert the `.Rnw` files to `.tex` files, run

```
./knit_iris_to_tex.sh
./knit_mice_data_to_tex.sh
./knit_structure_to_tex.sh
./knit_timing_table_to_tex.sh
```

which produce the files `experiments_iris.tex, experiments_mice.tex, experiments_structure.tex` and `timing_table.tex`, respectively. 
**Writing in these `.tex` files will be overwritten!**

**A note about caching**. The `./knit_*_to_tex.sh` commands are slow, as they are generating the figures. Thus, if the figures are not changing (i.e. only the text is being edited), set `simple_cache <- TRUE` in the first code chunk of the `.Rnw` files. Then, the first call of `./knit_*_to_tex.sh` will cache the figures (and it may be slow) --- but subsequent calls will use the cached figures and thus be fast. 
