## LaTeX conventions

- We keep text out of `main.tex`.

- Please use `refstyle` and `varioref` references.  The reference types
are all specified in `_reference_defs.tex`.

- Most mathematical symbols that are used repeatedly have macros
defined in `_math_defs.tex`.  When possible, please use these macros, since
that allows for easy search-and-replace.  This is especially true for
single Latin letters like `x` and `t`, which are otherwise very difficult to
rename with search-and-replace.

- I (Ryan) really like to use the `%` sign to improve (my perception of) the
readability of `tex`. Please indulge me!  In particuar I really like
well-wrapped text, and my text editors's auto-wrap tool will wrap math
environments together with text unless the math environment is surrounded by `%`
signs.  That's why almost all the align environments begin and end with a `%`
symbol.



## Knitr

The experiment files are all written using R knitr. They load in data from our experiments and create figures in code chunks in between text.

To download the necessary data for the knitr files, run
```
./R_scripts/sync_data_runjing.sh # pulls data from SCF /scratch/ folder
./R_scripts/process_data.sh # converts .npz files to .RData files
```
Note that syncing requires access to the Berkeley SCF cluster.  If you
don't have access, ask us for a zip file of the needed data.

The second step saves the data into the `processed_data` folder as `.RData` files, which the knitr files subsequently load.

**All writing for the experimental results is done in the .Rnw files**. The relevant knitr files for the experiments are:
- `experiments_iris.Rnw`
- `experiments_mice.Rnw`
- `experiments_structure.Rnw`
- and `timing_table.Rnw`


Then, to convert the `.Rnw` files to `.tex` files, run

```
./knit_everything.sh
```

which produces a `tex` files for every `Rnw` file.
**Writing in these `.tex` files will be overwritten!**

**A note about caching**. The `./knit_*_to_tex.sh` commands are slow, as they are generating the figures. Thus, if the figures are not changing (i.e. only the text is being edited), set `simple_cache <- TRUE` in the first code chunk of the `.Rnw` files. Then, the first call of `./knit_*_to_tex.sh` will cache the figures (and it may be slow) --- but subsequent calls will use the cached figures and thus be fast.

Alternatively, you can run the following commands to manually clear the cache:
```
rm figures/*
rm cache/*
```
After removing the content of these directories, knitting will regenerate
all the content, irrespetive of the value of `simple_cache`.
