# Cat Image Classification: Project Overview  
This project builds a deep learning model to automatically classify various cat (Felis) species from their images.

- Utilizing the Felis Taxonomy Image Classification dataset from Kaggle.
- Performing data preprocessing and image augmentation to enhance model generalization.
- Employing transfer learning technique on a pre-trained model for improved accuracy.
- Developing a user-friendly Flask API for real-time species prediction based on image inputs.


## Code and Resources Used 
**Python Version:** 3.12  
**Packages:** numpy, pandas, matplotlib, tensorflow, tensorflow_datasets, flask, pillow  
**Flask API Setup:**
- ```pip install -r requirements.txt```  
- ```conda env create -n <ENVNAME> -f environment.yaml``` (Anaconda Environment)
  
**Dataset:** https://www.kaggle.com/datasets/datahmifitb/felis-taxonomy-image-classification/data


## Getting Data
The project utilizes the <a href="https://www.kaggle.com/datasets/datahmifitb/felis-taxonomy-image-classification/data">Felis Taxonomy Image Classification</a> dataset, containing 519 JPG images of seven different cat species:

* Domestic cat (F. catus)
* European wildcat (F. silvestris)
* Jungle cat (F. chaus)
* African wildcat (F. lybica)
* Black-footed cat (F. nigripes)
* Sand cat (F. margarita)
* Chinese mountain cat (F. bieti)

![alt text](https://github.com/polaternez/felis-image-classification/blob/master/reports/figures/train_images.png "Train images")


## Data Preprocessing
- **Data Split:** The dataset was divided into 80% for training and 20% for testing to ensure the model learns well and is evaluated fairly.
- **Image Preprocessing:** All images were resized to 224x224 pixels so they are perfectly uniform for the computer to read.
- **Image Data Augmentation:** To make the model more adaptable and prevent overfitting (getting too memorized on the training images), these techniques were applied:
  - Image Rotation
  - Image Translation (shifting)
  - Image Flipping
  - Contrast Adjustment

## Model Building 
The brain of this project relies on EfficientNetB0, a highly powerful pre-trained image recognition model. We adapted it using transfer learning (taking a model that already knows how to see shapes and colors and teaching it specifically about cats).

Here is the blueprint of our network:

![alt text](https://github.com/polaternez/felis-image-classification/blob/master/reports/figures/model.png "Convolutional Neural Network(CNN)")


## Model Evaluation 
The model's performance is measured using categorical cross-entropy (a score of how confident it is with its guesses) and optimized using the ADAM algorithm (the math tool that helps the model learn from its mistakes).

The training results are shown below:

![alt text](https://github.com/polaternez/felis-image-classification/blob/master/reports/figures/model_evaluation.png "Model Performances")


## Productionization 
To bring the model to life, a user-friendly API was developed using Flask. This allows the model to act as a backend service: you upload a cat photo, and it instantly sends back the predicted species name.

![alt text](https://github.com/polaternez/felis-image-classification/blob/master/reports/figures/flask-api.png "Felis Image Classification API")







