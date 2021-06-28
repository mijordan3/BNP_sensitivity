# BNP sensitivity

This repo contains code to reproduce the experiments found in our manuscript _arXiv link coming soon_. 

Code to produce our paper can be found in the `writing/journal_paper` directory. A pdf can be produced by compiling the `main.tex` file, e.g. using `pdflatex main.tex`. 

### installation 

Install package used for our BNP model with: 
```
pip install BNP_modeling
```

dependencies include [jax](https://jax.readthedocs.io/en/latest/index.html) and the [jax branch of paragami](https://github.com/rgiordan/paragami/tree/jax). These will be install automatically with the command above. 

Our iris experiments, mice experiments, and population genetics experiments are contained in the `./GMM_clustering/`, `GMM_regression_clustering`, and `./structure/` folders, respectively. To install libraries specific to those experiments, run

```
pip install GMM_clustering
pip install GMM_regression_cluster
pip install structure
```
respectively. 

