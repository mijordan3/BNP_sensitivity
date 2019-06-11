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
pip install -e .
python3 -m ipykernel install --user --name=bnpregcluster_runjingdev
~~~

Then check out and install ``aistats2019_ij_paper`` from the (private) repo
[InfinitesimalJackknifeGenomicsExperiment](https://github.com/Runjing-Liu120/InfinitesimalJackknifeGenomicsExperiment).  Finally,

~~~
pip install -e .
python3 -m ipykernel install --user --name=bnpregcluster_runjingdev
~~~
