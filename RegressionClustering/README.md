This clusters the output of a regression using code from the
``aistats2019_ij_paper`` package in the repository
``https://github.com/Runjing-Liu120/InfinitesimalJackknifeGenomicsExperiment``.

To install, run

~~~
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy
pip install -r requirements.txt
~~~

Then check out and install ``aistats2019_ij_paper`` from the (private) repo
[InfinitesimalJackknifeGenomicsExperiment](https://github.com/Runjing-Liu120/InfinitesimalJackknifeGenomicsExperiment).  Note that as of writing, you need the ``moar_genez`` branch.

Also you have to install the other libraries in this repo.  From the base
of the ``BNP_sensitivity`` repsository, run

~~~
pip install GMM_clustering
pip install BNP_modeling
~~~

Finally, install the package in ``RegressionClustering`` and create a Jupyter notebook kernel for the virual environment.

~~~
pip install -e RegressionClustering
python3 -m ipykernel install --user --name=bnpregcluster_runjingdev
~~~
