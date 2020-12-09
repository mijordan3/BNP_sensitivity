The folder contains the Human Genome Diversity Panel data. 

The code to convert plink files to numpy requires Cython code. After installing Cython, build using

```
python setup.py build_ext --inplace
```

Then we download and process the HGDP data. To do this, run 
```
./get_hgdp_data.sh
```
