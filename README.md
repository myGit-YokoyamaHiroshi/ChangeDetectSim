# ChangeDetectSim
The GitHub repository associated with the peer-reviewed manuscript, entitled "Detecting changes in dynamical structures in synchronous neural oscillations using probabilistic inference" at NeuroImage, https://doi.org/10.1016/j.neuroimage.2022.119052 <br>
<br>
There is sample of python implementation for detecting the changes in the synchronous oscillatory networks, based on dynamical Bayesian inference. <br>
<br>
<img src="./figures/GraphicalAbstract.png" width=100%>

# Folder structures<br>
\ChangeDetectSim<br>
&ensp;&ensp;├── \figures <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;├─ \our_method … contains sample figures generated by the script (see sim_proposed_method.py) <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;└─ \VAR        … contains sample figures generated by the script (see sim_AR_Kalman.py) <br>
&ensp;&ensp;│<br>
&ensp;&ensp;├─ \my_modules<br>
&ensp;&ensp;│&ensp;&ensp;&ensp;├─ my_dynamical_bayes_mod.py … contains main modules of our proposed change-detection method <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;├─ ar_kalman_connectivity.py … contains main modules of VAR-based network change-detection method <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;├─ my_oscillator_model.py    … contains some functions for synthetic data generation <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;└─ my_graph_visualization.py … contains some functions for the figure visualization results <br>
&ensp;&ensp;│<br>
&ensp;&ensp;├── \save_data <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;└─ \param_sim\Sim_param.npy <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;… contains the sample parameter settings <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; &ensp; &ensp; to reproduce the simulation results performed in the original paper <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; &ensp; &ensp;  &ensp; (Yokoyama & Kitajo, NeuroImage, 2022)<br>
&ensp;&ensp;│<br>
&ensp;&ensp;├─ sim_proposed_method.py <br>
&ensp;&ensp;└─ sim_AR_Kalman.py <br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;… contains sample scripts to reproduce the simulation results performed in the original paper   <br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;  (Yokoyama & Kitajo, NeuroImage, 2022)<br>
<br>
&ensp;*** For the implementation details, refer to paper, available at: https://doi.org/10.1016/j.neuroimage.2022.119052

# Requirements<br>
&ensp; Operation has been confirmed only under the following environment. <br>
&ensp;&ensp; - OS: Windows 10 64bit, Ubuntu 18.04.5 LTS <br>
&ensp;&ensp; - Python 3.8.3, 3.8.5 <br>
&ensp;&ensp; - IPython 7.16.1, 7.19.0 <br>
&ensp;&ensp; - conda 4.8.4, 4.9.2  <br>
&ensp;&ensp; - Spyder 4.1.4, 4.1.5 <br>
&ensp;&ensp; - numpy 1.18.5, 1.19.2 <br>
&ensp;&ensp; - scipy 1.5.0, 1.5.2 <br>
&ensp;&ensp; - matplotlib 3.2.2, 3.3.2<br>
&ensp;&ensp; - scikit-learn 0.23.1, 0.23.2 <br>
&ensp;&ensp; - joblib 0.16.0, 0.17.0 <br>
&ensp; <br>
&ensp; The sample scripts are not guaranteed to run on any other version in Python than the above.<br>
&ensp; <br>
# Authors<br>
&ensp; Hiroshi Yokoyama<br>
&ensp;&ensp;(Division of Neural Dynamics, Department of System Neuroscience, National Institute for Physiological Sciences, Japan)<br>

# References<br>
Yokoyama Hiroshi and Kitajo Keiichi. (2022). Detecting changes in dynamical structures in synchronous neural oscillations using probabilistic inference. NeuroImage, 119052. doi: 10.1016/j.neuroimage.2022.119052

# Cite<br>

Please cite our paper if you use this code in your own work:
```
@article{Yokoyama2022,
author = {Yokoyama, Hiroshi and Kitajo, Keiichi},
doi = {10.1016/j.neuroimage.2022.119052},
issn = {10538119},
journal = {NeuroImage},
month = {mar},
number = {March},
pages = {119052},
title = {{Detecting changes in dynamical structures in synchronous neural oscillations using probabilistic inference}},
url = {https://doi.org/10.1016/j.neuroimage.2022.119052},
volume = {252},
year = {2022}
}
```
```
@software{ChangeDetectSim,
  author = {Yokoyana H.},
  title = {ChangeDetectSim},
  url = {https://github.com/myGit-YokoyamaHiroshi/ChangeDetectSim},
  year = {2020}
}
```
