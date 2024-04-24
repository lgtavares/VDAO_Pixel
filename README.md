# Pixel-Based Change Detection in Moving-Camera Videos Using Twin Convolutional Features on a Data-Constrained Scenario

* [Overview](#overview)  
* [Requirements](#requirements)  
* [Project Usage](#usage)  
* [Results](#results)  


<a name="overview"></a>
## Overview

The proposed anomaly detection framework is depicted in the figure below:

![Abandoned object proposed framework](https://github.com/lgtavares/VDAO_Pixel/assets/4022337/4cd0955f-096e-4ce3-8a39-34601b7a21d5)

<a name="requirements"></a>
## Requirements

### Installing packages and libraries:

If you use conda/anaconda, use the file environment.yml to install the needed packages to run the scripts of the project:
`conda env create -f environment.yml`

After that, to put the VDAO_Pixel package in the newly created environment, run:
`python setup.py develop`

<a name="usage"></a>
## Project Usage

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

<a name="results"></a>
## Results

Below one can see the comparative tables between the previous state-of-the-art algorithms and the proposed PBCD-MC in the frame-level, in the object-level and in the pixel-level. The metrics used for such comparisons are the true-positive rate(TPR) and false-positive rate (FPR); the euclidean distance to the perfect classifier (DIS); and the Matthews Correlation Coefficient (MCC).

### TPR and FPR results 

#### Frame-level

|   | **DAOMC[[1]](#1)**      | **ADMULT[[2]](#2)**    | **MCBS[[3]](#3)**    | **mcDTSR[[4]](#4)**     | **TCF-LMO[[5]](#5)**        | **PBCD-MC**    |
|:-------:|:--------------:|:-------------:|:-----------:|:--------------:|:------------------:|:--------------:|
| **TPR** | 0.513          | 0.750         | 0.994       | **1.000** | 0.974              | 0.863          |
| **FPR** | **0.149** | 0.329         | 0.993       | 1.000          | 0.744              | 0.225          |


#### Object-level

|   | **DAOMC[[1]](#1)**      | **ADMULT[[2]](#2)**    | **MCBS[[3]](#3)**    | **mcDTSR[[4]](#4)**     | **TCF-LMO[[5]](#5)**        | **PBCD-MC**    |
|:-------:|:--------------:|:-------------:|:-----------:|:--------------:|:------------------:|:--------------:|
| **TPR** | 0.503          | 0.705         | 0.217       | 0.832          | **0.958**     | 0.845          |
| **FPR** | **0.206** | 0.657         | 0.997       | 1.000          | 0.906              | 0.341          |



#### Pixel-level

|   | **DAOMC[[1]](#1)**      | **ADMULT[[2]](#2)**    | **MCBS[[3]](#3)**    | **mcDTSR[[4]](#4)**     | **TCF-LMO[[5]](#5)**        | **PBCD-MC**    |
|:-------:|:--------------:|:-------------:|:-----------:|:--------------:|:------------------:|:--------------:|
| **TPR** | 0.675          | 0.651         | 0.025       | 0.580          | 0.809              | **0.907** |
| **FPR** | 0.005          | 0.005         | 0.016       | 0.423          | 0.020              | **0.004** |

### DIS results 

#### Frame-level

|   | **DAOMC[[1]](#1)**      | **ADMULT[[2]](#2)**    | **MCBS[[3]](#3)**    | **mcDTSR[[4]](#4)**     | **TCF-LMO[[5]](#5)**        | **PBCD-MC**    |
|:-------:|:--------------:|:-------------:|:-----------:|:--------------:|:------------------:|:--------------:|
| **Average** | 0.284             | 0.353          | 0.003        | 0.000          | 0.223              | **0.546**      |
| **Median**  | 0.269             | 0.511          | 0.000        | 0.000          | 0.000              | **0.762**      |
| **Overall** | 0.327             | 0.388          | 0.004        | 0.000          | 0.360              | **0.638**      |

#### Object-level

|   | **DAOMC[[1]](#1)**      | **ADMULT[[2]](#2)**    | **MCBS[[3]](#3)**    | **mcDTSR[[4]](#4)**     | **TCF-LMO[[5]](#5)**        | **PBCD-MC**    |
|:-------:|:--------------:|:-------------:|:-----------:|:--------------:|:------------------:|:--------------:|
| **Average** | 0.259             | 0.086          | -0.849       | -0.249         | 0.126              | **0.485**      |
| **Median**  | 0.247             | 0.038          | -0.950       | -0.131         | 0.000              | **0.737**      |
| **Overall** | 0.271             | 0.050          | -0.826       | -0.323         | 0.103              | **0.517**      |

#### Pixel-level


|   | **DAOMC[[1]](#1)**      | **ADMULT[[2]](#2)**    | **MCBS[[3]](#3)**    | **mcDTSR[[4]](#4)**     | **TCF-LMO[[5]](#5)**        | **PBCD-MC**    |
|:-------:|:--------------:|:-------------:|:-----------:|:--------------:|:------------------:|:--------------:|
| **Average** | 0.423             | 0.538          | 0.003        | 0.016          | 0.570              | **0.724**      |
| **Median**  | 0.454             | 0.622          | -0.006       | -0.010         | 0.607              | **0.855**      |
| **Overall** | 0.699             | 0.688          | 0.011        | 0.047          | 0.612              | **0.877**      |


### MCC results 

#### Frame-level

|   | **DAOMC[[1]](#1)**      | **ADMULT[[2]](#2)**    | **MCBS[[3]](#3)**    | **mcDTSR[[4]](#4)**     | **TCF-LMO[[5]](#5)**        | **PBCD-MC**    |
|:-------:|:--------------:|:-------------:|:-----------:|:--------------:|:------------------:|:--------------:|
| **Average** | 0.559         | 0.434         | 0.860       | 0.864         | 0.672              | **0.324**  |
| **Median**  | 0.539         | 0.295         | 1.000       | 1.000         | 0.872              | **0.152**  |
| **Overall** | 0.509         | 0.413         | 0.993       | 1.000         | 0.744              | **0.264**  |

#### Object-level

|   | **DAOMC[[1]](#1)**      | **ADMULT[[2]](#2)**    | **MCBS[[3]](#3)**    | **mcDTSR[[4]](#4)**     | **TCF-LMO[[5]](#5)**        | **PBCD-MC**    |
|:-------:|:--------------:|:-------------:|:-----------:|:--------------:|:------------------:|:--------------:|
| **Average** | 0.570         | 0.716         | 1.311       | 1.047         | 0.840              | **0.402**  |
| **Median**  | 0.539         | 0.768         | 1.366       | 1.000         | 1.000              | **0.189**  |
| **Overall** | 0.538         | 0.721         | 1.268       | 1.014         | 0.907              | **0.374**  |

#### Pixel-level

|   | **DAOMC[[1]](#1)**      | **ADMULT[[2]](#2)**    | **MCBS[[3]](#3)**    | **mcDTSR[[4]](#4)**     | **TCF-LMO[[5]](#5)**        | **PBCD-MC**    |
|:-------:|:--------------:|:-------------:|:-----------:|:--------------:|:------------------:|:--------------:|
| **Average** | 0.586         | 0.516         | 0.985       | 0.738         | 0.239              | **0.221**  |
| **Median**  | 0.500         | 0.514         | 0.999       | 0.769         | 0.144              | **0.062**  |
| **Overall** | 0.325         | 0.348         | 0.975       | 0.596         | 0.192              | **0.092**  |


## References
<a id="1">[1]</a> 
H. Kong, J.-Y. Audibert, J. Ponce, Detecting abandoned objects with a moving camera, IEEE Transactions on Image Processing 19 (8) (2010) 2201–2210. doi:10.1109/TIP.2010.2045714.

<a id="2">[2]</a> 
G. H. F. de Carvalho, L. A. Thomaz, A. F. da Silva, E. A. B. da Silva, S. L. Netto, Anomaly detection with a moving camera using multi-scale video analysis, Multidimensional Systems and Signal Processing 30 (2019) 311–342.

<a id="3">[3]</a> 
H. Mukojima, D. Deguchi, Y. Kawanishi, I. Ide, H. Murase, M. Ukai, N. Nagamine, R. Nakasone, Moving camera background-subtraction for obstacle detection on railway tracks, in: Proceedings of the IEEE 28 International Conference on Image Processing, 2016, pp. 3967–3971. doi:10.1109/ICIP.2016.7533104.


<a id="4">[4]</a> 
E. Jardim, L. A. Thomaz, E. A. B. da Silva, S. L. Netto, Domain-transformable sparse representation for anomaly detection in moving-camera videos, IEEE Transactions on Image Processing 29 (2020) 1329–1343. doi:10.1109/TIP.2019.2940686.


<a id="5">[5]</a> 
R. Padilla, A. F. da Silva, E. A. da Silva, S. L. Netto, Change detection in moving-camera videos with limited samples using twin-CNN features and learnable morphological operations, Signal Processing: Image Communication 115 (2023) 116969.
