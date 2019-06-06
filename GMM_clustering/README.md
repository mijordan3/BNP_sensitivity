
To create a virtual environment and install the package, create an empyt directory ``venv`` and run

~~~
python3 -m venv venv
source venv/bin/activate
pip install -e .
python3 -m ipykernel install --user --name=bnpgmm_runjingdev
~~~

To delete this kernel when you are through, you can run ``jupyter kernelspec list`` and delete the directory corresponding to the ``bnpgmm_runjingdev`` kernel.
