# Particle Classification Model using Higgs Dataset

## Overview

This project implements a deep learning model for classifying particle collision events into two categories: **Signal** and **Background**. The model is trained on the **Higgs Boson dataset** and can predict whether a particle event is related to a Higgs boson signal or background noise based on several input features. The model is built using **PyTorch** and is capable of making predictions on custom input data after training.

## Dataset

The **Higgs Boson dataset** contains simulated particle collision events and is used for binary classification. The dataset includes various features of the particles involved in the collisions. The dataset is publicly available on the **UCI Machine Learning Repository**.

### Dataset Parameters (Features)

The dataset contains the following features:

1. **label**: Target variable (0 = Background, 1 = Signal)
2. **lepton_pT**: Transverse momentum of the lepton (float)
3. **lepton_eta**: Pseudorapidity of the lepton (float)
4. **lepton_phi**: Azimuthal angle of the lepton (float)
5. **missing_energy_magnitude**: Magnitude of missing energy in the event (float)
6. **missing_energy_phi**: Azimuthal angle of missing energy (float)
7. **jet_1_pt**: Transverse momentum of the first jet (float)
8. **jet_1_eta**: Pseudorapidity of the first jet (float)
9. **jet_1_phi**: Azimuthal angle of the first jet (float)
10. **jet_1_b-tag**: b-tagging score for the first jet (float)
11. **jet_2_pt**: Transverse momentum of the second jet (float)
12. **jet_2_eta**: Pseudorapidity of the second jet (float)
13. **jet_2_phi**: Azimuthal angle of the second jet (float)
14. **jet_2_b-tag**: b-tagging score for the second jet (float)
15. **jet_3_pt**: Transverse momentum of the third jet (float)
16. **jet_3_eta**: Pseudorapidity of the third jet (float)
17. **jet_3_phi**: Azimuthal angle of the third jet (float)
18. **jet_3_b-tag**: b-tagging score for the third jet (float)
19. **jet_4_pt**: Transverse momentum of the fourth jet (float)
20. **jet_4_eta**: Pseudorapidity of the fourth jet (float)
21. **jet_4_phi**: Azimuthal angle of the fourth jet (float)
22. **jet_4_b-tag**: b-tagging score for the fourth jet (float)
23. **m_jj**: Invariant mass of the two leading jets (float)
24. **m_jjj**: Invariant mass of the three leading jets (float)
25. **m_lv**: Invariant mass of the lepton and missing energy (float)
26. **m_jlv**: Invariant mass of the jet, lepton, and missing energy (float)
27. **m_bb**: Invariant mass of the two b-tagged jets (float)
28. **m_wbb**: Invariant mass of the Higgs-like system formed from jets and missing energy (float)
29. **m_wwbb**: Invariant mass of the four jets (float)

The dataset includes a large number of samples, and each sample represents a particle collision event.

## Model Architecture

The model is a **fully connected feedforward neural network** implemented using **PyTorch**. The architecture consists of the following layers:

1. **Input Layer**: The number of input neurons corresponds to the number of features in the dataset (29 features).
2. **Hidden Layer 1**: 128 neurons with ReLU activation.
3. **Hidden Layer 2**: 64 neurons with ReLU activation.
4. **Output Layer**: 2 neurons (binary classification: Signal vs Background) with a softmax activation function.

The model is trained using the **CrossEntropyLoss** as the loss function and the **Adam** optimizer.

## Usage

### 1. **Training the Model**

To train the model on the Higgs Boson dataset, you can run the provided code that reads the dataset, preprocesses it, and trains the neural network. The training process will optimize the model to classify particle collision events as either **Signal** or **Background**.

1. Load and preprocess the dataset.
2. Train the model on the training data.
3. Evaluate the model on the test data.
4. Save the trained model and scaler to Google Drive.

### 2. **Making Predictions**

After the model is trained and saved, you can use it for custom predictions. You can input the particle features (like lepton transverse momentum, jet information, etc.) and the model will classify the particle as **Signal** or **Background**.


### 3. **Input Format for Prediction**

To make a prediction, input the following particle features:

1. Lepton pT (float)
2. Lepton eta (float)
3. Lepton phi (float)
4. Missing energy magnitude (float)
5. Missing energy phi (float)
6. Jet 1 pT (float)
7. Jet 1 eta (float)
8. Jet 1 phi (float)
9. Jet 1 b-tag (float)

## Dependencies

The following Python packages are required to run this project:

- torch
- numpy
- pandas
- scikit-learn
- matplotlib (optional, for visualization)
- pickle

You can install the required dependencies by running:

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## Results

After training the model, you will get a **classification report** that includes metrics such as **precision**, **recall**, **f1-score**, and **accuracy**. These metrics will help you evaluate how well the model is performing.

## Conclusion

This model is designed to classify particle collision events as either **Signal** or **Background**. It is trained using the **Higgs Boson dataset** and can be used for real-time predictions on new input data. By using this model, researchers can classify events more effectively in particle physics experiments.

## License

This project is licensed under the MIT License.
