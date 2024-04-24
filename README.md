# Pixel-Based Change Detection in Moving-Camera Videos Using Twin Convolutional Features on a Data-Constrained Scenario
------------

* [Overview](#overview)  

The proposed anomaly detection framework is depicted in the figure below:

<p align="center">
<img src="https://github.com/lgtavares/VDAO_Pixel/blob/master/doc/images/graphical_abstract.png?raw=true" align="center"/></p>

![Abandoned object proposed framework](doc/images/graphical_abstract.png?raw=true)

* [Requirements](#requirements)  

### Installing packages and libraries:

If you use conda/anaconda, use the file environment.yml to install the needed packages to run the scripts of the project:
`conda env create -f environment.yml`

After that, to put the VDAO_Pixel package in the newly created environment, run:
`python setup.py develop`


* [Project Usage](#usage)  

The main project code is organized in the following manner:

- *data/*  - Store important data files like alignment tables, feature tensors, results of intermediary modules, temporary classifiers, etc.
- *doc/*   - Store important files for documentation.
- *extra/* - Store project related material, like codes from other sources for results comparisson, etc.
- *models/* - Store the project resulting classifier modules.
- *notebooks/* - Store some Jupyter Notebooks for module testing.
- *results/* - Store the result tables, images and videos for every framework configuration.
- *scripts/* - Store the scripts for each framework module.
    - *alignment/* - Scripts for the alignment module;
    - *classification/*
    - *extra/*
    - *features/*
    - *postprocessing/*
    - *results/*
    - *test/*
- *src/* - Store the main code with the classes, functions and libraries needed to run rhe scripts.
* [Results](#results)  

