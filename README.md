# JetPointNet

## Overview

JetPointNet adapts the PointNet architecture to enhance energy measurement precision in particle physics. This project processes CERN Root files into machine learning-compatible formats, enabling sophisticated analysis of collision events from the ATLAS detector.

## Key Components

- **Data Preprocessing Pipeline**: Converts Root files to Awkward Arrays, then to NumPy format
- **Model Architecture**: Customized PointNet with physics-inspired modifications
- **Training & Evaluation**: Specialized scripts for model training and performance assessment

## Project Structure

JetPointNet/\
├── python_scripts/\
│   └── data_processing/\
│       ├── jets/\
│       │   ├── jets_awk_to_npz.py\
│       │   ├── jets_root_to_awk.py\
│       │   ├── preprocessing_header.py\
│       │   ├── track_metadata.py\
│       │   └── util_functs.py\
│       └── jets_training/\
│           ├── models/\
│           ├── jets_test.ipynb\
│           ├── jets_train.py\
│           └── jets_tune.py\
└── README.md

## Requirements

- Python 3.8+
- TensorFlow 2.5.0+
- CUDA 11.1+
- Awkward Array 1.6.0+
- Uproot 4.0.0+
- NumPy 1.20.0+

## Setup

1. Acquire Athena Events Dataset
2. Clone the repository:
3. Install dependencies:
4. Configure CUDA environment (if using GPU)

## Usage

1. Preprocess data step 1 (Root to Awkward Array):
run `python_scripts/data_processing/jets/jets_root_to_awk.py`

3. Preprocess data step 2 (Awkward Array to Numpy):
run `python_scripts/data_processing/jets/jets_awk_to_npz.py`

4. Train model:
run `python_scripts/data_processing/jets_training/jets_train.py`

5. Evaluate model:
Run `python_scripts/data_processing/jets_training/jets_test.ipynb`

## Results

Preliminary tests demonstrate promising qualitative performance of around 73% in attributing significant energy deposits within the detector to specific source particles.

## Future Work

- Implement hyperparameter optimization
- Explore attention mechanisms for improved performance
- Scale the model to handle larger datasets

## Contributing

We welcome contributions to the JetPointNet project, feel free to fork away!

## Acknowledgements

We gratefully acknowledge the support of TRIUMF, CERN and the ATLAS group in providing the data and infrastructure necessary for this research.

Note:
Please note that the onboarding documentation is a bit out of date (as of 2024/04/26)
