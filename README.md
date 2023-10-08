# NASA Space Apps Challenge 2023 - Building the Space Biology “Model Zoo”
[Learn More](https://www.spaceappschallenge.org/2023/challenges/building-the-space-biology-model-zoo/)


## Project Description

Convolutional model that uses a database of images of viruses and bacteria, obtained from public databases, which predicts which group the images belong to.

## Table of Contents

- [Data](#data)
- [Installation](#installation)
- [Python Code](#code)
- [Usage](#usage)
- [Contributors](#contributors)
- [Resources](#resources)

## Data

### Microorganism Database (Microorganism.zip)

- Description: A database of viruses and bacteria images compiled from publicly available sources. The images have been preprocessed and organized in order for them to be used in the project.
- Total Images: 982 (492 bacteria, 490 virus)
- Data Split: 70% for training, 30% for testing
- Directory Structure:
  - 'microorganisms': 0 images
  - 'microorganisms\test': 0 images
  - 'microorganisms\test\bacteria': 98 images
  - 'microorganisms\test\virus': 98 images
  - 'microorganisms\train': 0 images
  - 'microorganisms\train\bacteria': 394 images
  - 'microorganisms\train\virus': 392 images

### Prediction Dataset (Predecir.zip)

- Description: A subset of the test data from the Microorganism Database used for manual predictions. It contains two folders, one for viruses and one for bacteria, with images numbered for reference.

## Installation

To run this project locally, you will need Python version 3.10.12 or later, along with the following dependencies:

- [Matplotlib](https://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

Follow these steps to set up your local environment:

1. **Python Installation**: If you don't have Python 3.10.12 or later installed, you can download it from the [official Python website](https://www.python.org/downloads/).

2. **Dependency Installation**: Open a terminal or command prompt and run the following commands to install the required Python libraries:

   ```bash
   pip install matplotlib
   pip install tensorflow
   pip install pandas
   pip install numpy
## Model
### Python Code (Conv_ModelZoo.ipynb)
It is a python code designed during the event (October 7, 2023 and October 8, 2023), which is responsible for decompressing the .zip of microorganisms and extracting it. Subsequently, it shows its composition, modifies the training images for better performance, normalizes them to reduce the computational load, organizes them in batches and applies the model. Finally, take a previously specified image from the predict folder to use the network practically.

## Usage
To make predictions using the Convolutional Neural Network, follow these steps:

1. **Installation**: Ensure that you have installed Python 3.10.12 or later, along with the required dependencies as explained in the [Installation](#installation) section.
2. **Modifications**: To predict with the Convolutional Neural Network it is necessary to replace the code section:
   "# Ruta de la imagen que deseas predecir
imagen_ruta = "Predecir/Bacteria/Prueba (4).jpg"  # Ajusta la ruta según la ubicación real"
With the real path of the image location to use.
3. **Display** The image used will be displayed with a header indicating the group that it belongs to (virus or bacteria).
## Contributors
- [Ninoska Setien](https://github.com/ninosetien1804) -programmer
- [Abigail Perrusquia](link_to_profile_2) - programmer
- [Saul Troncoso](link_to_profile_2) - Dataset
- [Julio Cruz](link_to_profile_2) - Dataset
- [Cecilia González](link_to_profile_2) -Dataset
- [Pablo Hernandez](link_to_profile_2) - Dataset & programmer
- [Marcos Avilés](link_to_profile_2) - Assesor 
## Resources

- [Link to Kaggle Dataset](https://www.kaggle.com/datasets/mdwaquarazam/microorganism-image-classification)
- [Doi to Kaggle Dataset](https://doi.org/10.34740/KAGGLE/DSV/4032122)
- [Link to Kaggle Dataset] https://www.kaggle.com/datasets/saurabhshahane/virus-images
- [support link for deep learning](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/04_transfer_learning_in_tensorflow_part_1_feature_extraction.ipynb)

