<h1>PointNet for particle segmentation:</h1>
(Note: this repo contains PointNet for particle segmentation and EM frac learning) </br>

<h3>Data processing:</h3>

Processing outlined in https://github.com/jessicabohm/PointNet_Segmentation/tree/ml1_train/python_scripts/data_processing/particle_deposit_learning

<h3>Training model:</h3>

First train was on rho dataset with:
* Train notebook https://github.com/jessicabohm/PointNet_Segmentation/blob/ml1_train/python_scripts/train_pnet.ipynb that trains on rho event
* I've been mainly using https://github.com/jessicabohm/PointNet_Segmentation/blob/ml1_train/python_scripts/pnet_cell_classification.py script but there is some additional data processing mixed into the training script </br></br>


More up-to-date training scripts and analysis for delta dataset:
* Train script https://github.com/jessicabohm/PointNet_Segmentation/blob/ml1_train/python_scripts/train_pnet_delta.py or notebook https://github.com/jessicabohm/PointNet_Segmentation/blob/ml1_train/python_scripts/train_pnet_delta.ipynb
* Analysis of delta train is in https://github.com/jessicabohm/PointNet_Segmentation/blob/ml1_train/nbs/particle_deposit_learning/delta/delta_train_analysis.ipynb