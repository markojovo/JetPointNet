# JetPointNet 🚀🔬

Revolutionizing jet energy reconstruction in the ATLAS detector using deep learning.

## 🌟 Overview

JetPointNet adapts the PointNet architecture for precise energy measurement in particle physics. It processes CERN Root files into machine-learning-ready formats, enabling advanced analysis of collision events.

## 🛠 Key Components

- **Data Preprocessing**: Root → Awkward → Numpy pipeline
- **Model Architecture**: Custom PointNet with physics-inspired modifications
- **Training & Evaluation**: Specialized scripts for model training and performance assessment

## 🔧 Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Configure CUDA environment (if using GPU)

## 🚀 Usage

1. Preprocess data: `python jets_root_to_awk.py`
2. Convert to NumPy: `python jets_awk_to_npz.py`
3. Train model: `python jets_train.py`
4. Evaluate: Run `jets_test.ipynb`

## 📊 Results

Preliminary tests show promising qualitative performance in identifying important energy deposits.

## 🔮 Future Work

- Hyperparameter optimization
- Exploring attention mechanisms
- Scaling to larger datasets


Please note that the onboarding documentation is a bit out of date (as of 2024/04/26)
