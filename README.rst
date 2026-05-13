==========
Echoedge
==========

|doi| |license| |OpenSSF Best Practices|

.. |OpenSSF Best Practices| image:: https://www.bestpractices.dev/projects/9547/badge
   :target: https://www.bestpractices.dev/projects/9547

:Version: 1.0.0
:Download: https://github.com/liliaguillet/echoedge.git
:Source: https://github.com/liliaguillet/echoedge.git
:Keywords: scientific software, fish schools, echosounder,
    python

Echoedge is a repository with code and instructions on how to run echogram processing and analysis with Python. 

.. image:: Drone2.png
   :align: center

Installation
============

From source
-----------

Prerequisites: Check your Python version
+++++++++++++
Start by checking your current environment, our configurations are shown below.
It should be possible to run with other python-versions and other operating systems. 
Continue by cloning this repo, installing necessary packages and creating a cronjob. 
Please note that the latest version of Echopype is only compatible with Python>=3.9.


To get the source, clone the last version of Echoedge repository:

.. code-block::

   git clone https://github.com/liliaguillet/echoedge.git
   cd echoedge

Install the software
++++++++++++++++++++

First, install the dependencies and create the environment:

.. code-block:: 
   python3 -m venv venv
   source venv/bin/activate
   pip3 install -r requirements.txt
   pip3 install -e . # install library for this repo

Then, build and install the software:


Bug tracker
===========

If you have any suggestions, bug reports, or annoyances please report
them to our issue tracker at https://github.com/.../issues.

Structure
========
echoedge/
├── lib/
├── postprocessing/
│   └── subfolder-per-survey/
├── test/
├── .gitignore
├── requirements.txt
└── README.rs

Acknowledgements
================
The processing of the raw-files from the echosounder is based on the **Echopype**(https://echopype.readthedocs.io/en/stable/) library.  


Citation
========

As part of the research process, it is important that pieces of software
that have contributed to the research are cited. The source code is available on GitHub[1] and archived in Zenodo[2].

#. https://github.com/ ... 

#. Guillet, M., Liu, C. François, Y. & Hentati Sundberg, J.Echoedge v1.0.0.
   *Zenodo* **2022**, https://doi.org/


License
=======

Echoedge is released under the Apache 2.0 license, as found
in the `LICENSE <LICENSE>`_ file.

.. |DOI| image:: https://zenodo.org/badge/DOI/ ... 
   :target: https://doi.org/ .... 

.. |license| image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
    :alt: Apache 2.0 
    :target: https://opensource.org/licenses/Apache-2.0


