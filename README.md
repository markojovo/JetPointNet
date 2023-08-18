<h1>PointNet for particle segmentation:</h1>
(Note: this repo contains PointNet for particle segmentation and EM frac learning) </br>

<h3>Data processing:</h3> 

* Input: unprocessed event root files - either single delta or rho events
* Run "process_root_to_train_data.py process_root_to_train_data_config.py"
* PointNet_Segmentation/python_scripts/data_processing/particle_deposit_learning/<b>process_root_to_train_data_config.yaml</b>
* PointNet_Segmentation/python_scripts/data_processing/particle_deposit_learning/<b>preprocess_events_to_pnet_npz_delta.py</b></br></br>

This will save two folders of event data to save_dir:
* <b>processed_train_files</b> - npz files with X and Y train data to pass to the model for training and evaluation
* <b>processed_test_files</b> - npy files of all events and all features just with some of their features preprocessed  (saved since processing from root to preprocessed events npy is quite slow - particularly the flattening of clusters to events), load this data for analysis instead of loading from root<br><br>

<b>Processing workflow:</b></br>
Preprocessing before splitting by events:
1. Cut out negative energy deposits - a small percentage (0.03% in the rho dataset) of cluster_cell_hitsTruthE are negative => cut these out. Note: if the cell only had negative hits it will now have 0 hits
2. Cut out the padding from the end of the particle deposits arrays. Most cell features have been cut down to len of cells with E > threshold, however, cluster_cell_hitsTruthE and cluster_cell_hitsTruthIndex have not been => cut out this padding 
3. Flatten clusters to events and remove the cells that are repeated in multiple topoclusters
4. If processing delta dataset label each event with its decay group
5. Save the preprocessed npy files to processed_test_files
	
For each event, 
1. Get num tracks - and discard event if num tracks doesn’t match its decay groups num tracks (ie. Delta event with 2 charged particles => 2 tracks, rho event => 1 track)
2. Compute the cartesian points of the track hits - one per layer of calorimeter => 2D array of track hits per event (num_tracks by NUM_TRACK_POINTS_NUM)
3. Get cell E and x, y, z of all cells in event
4. Get cell labels dependent on decay group
    * For rho events label cells 1 if the neutral pion deposited more energy in the cell and 0 if the charged pion deposited the majority of the cells energy
    * For delta events label point 0 - particle associated with track of interest, 1 - other tracked particles, 2 - neutral pion, or 3 - other neutral hadrons
5. If delta event with 2 tracks - match the track to the particle closest & threshold that both tracks must be close enough to the particle they are matched with - if no matching exists with close enough pairings discard event
6. Process the data to PointNet model train input
    * X data is event_point_data [num events x num points x num features] with features = (cell E, x, y, z, type of point label) for cell points and (track P, x, y, z, type of point label) for track points, where type of point label is 0-point, 1-track of interest, 2-other track (2 only for delta dataset)
    * Y data is binary class for rho events (0-pi0, 1-pipm) or 4 classes for delta dataset (0-charged of interest, 1-other charged, 2-pi0, 3-other neutral)<br><br>

<h3>Training model:</h3>

* Train notebook https://github.com/jessicabohm/PointNet_Segmentation/blob/ml1_train/python_scripts/train_pnet.ipynb that trains on rho event
* I've been mainly using https://github.com/jessicabohm/PointNet_Segmentation/blob/ml1_train/python_scripts/pnet_cell_classification.py script but there is some additional data processing mixed into the training script
