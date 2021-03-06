# ChangeDetectSim
This is sample of python implementation for detecting the changes in the synchronous oscillatory networks, based on dynamical Bayesian inference. <br>

# Folder structures<br>
\ChangeDetectSim<br>
&ensp;&ensp;├── \figures <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;├─ \sim1 … contains sample figures generated by the script (bayesian_oscillator_sim1.py) <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;├─ \sim2 … contains sample figures generated by the script (bayesian_oscillator_sim2.py) <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;└─ \sim3 … contains sample figures generated by the script (bayesian_oscillator_sim3.py) <br>
&ensp;&ensp;│<br>
&ensp;&ensp;├─ \my_modules<br>
&ensp;&ensp;│&ensp;&ensp;&ensp;├─ my_dynamical_bayes.py … contains main modules of dynamical Bayesian inference<br>
&ensp;&ensp;│&ensp;&ensp;&ensp;└─ my_graph_visualization.py … contains some function to visualize the estimation results<br>
&ensp;&ensp;│<br>
&ensp;&ensp;├── \save_data <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;├─ \param_sim1\Sim_param_sim1.npy <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;├─ \param_sim2\Sim_param_sim2.npy <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;└─ \param_sim3\Sim_param_sim3.npy <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;… contains the sample parameter settings <br>
&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; &ensp; &ensp; to reproduce the simulation results performed in the original paper (Yokoyama & Kitajo, bioRxiv, 2020) <br>
&ensp;&ensp;│<br>
&ensp;&ensp;├─ bayesian_oscillator_sim1.py <br>
&ensp;&ensp;├─ bayesian_oscillator_sim2.py <br>
&ensp;&ensp;└─ bayesian_oscillator_sim3.py <br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;… contains sample examples for how to use of the module “my_dynamical_bayes_mod.py”  <br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;… (These scripts reproduce the simulation results performed in the original paper (Yokoyama & Kitajo, bioRxiv, 2020) )<br>
<br>
&ensp;*** For the implementation details, refer to paper (preprint), available at: https://doi.org/10.1101/2020.10.13.335356

# Requirements<br>
&ensp; Operation has been confirmed only under the following environment. <br>
&ensp;&ensp; - Python 3.8.3 <br>
&ensp;&ensp; - conda 4.8.4  <br>
&ensp;&ensp; - Spyder 4.1.4 <br>
&ensp;&ensp; - numpy 1.18.5 <br>
&ensp;&ensp; - scipy 1.5.0 <br>
&ensp;&ensp; - matplotlib 3.2.2<br>
&ensp;&ensp; - networkx 2.4 <br>
&ensp;&ensp; - IPython 7.16.1 <br>
&ensp; <br>
&ensp; The sample scripts are not guaranteed to run on any other version in Python than the above.<br>
&ensp; <br>
# Authors<br>
&ensp; Hiroshi Yokoyama<br>
&ensp;&ensp;(Division of Neural Dynamics, Department of System Neuroscience, National Institute for Physiological Sciences, Japan)<br>

# References<br>
Hiroshi Yokoyama, Keiichi Kitajo, “Detection of the changes in dynamical structures in synchronous neural oscillations from a viewpoint of probabilistic inference”, bioRxiv (2020); doi: 10.1101/2020.10.13.335356
