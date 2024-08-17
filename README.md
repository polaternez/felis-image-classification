# Felis Image Classification: Project Overview  
This project builds a deep learning model to automatically classify various Felis species from their images.

- Utilizing the Felis Taxonomy Image Classification dataset from Kaggle.
- Performing data preprocessing and image augmentation to enhance model generalization.
- Employing fine-tuning techniques on a pre-trained model for improved accuracy.
- Developing a user-friendly Flask API for real-time species prediction based on image inputs.


## Code and Resources Used 
**Python Version:** 3.10  
**Packages:** numpy, pandas, matplotlib, tensorflow, tensorflow_datasets, flask, pillow  
**Flask API Setup:**
- ```pip install -r requirements.txt```  
- ```conda env create -n <ENVNAME> -f environment.yaml``` (Anaconda Environment)
  
**Dataset:** https://www.kaggle.com/datasets/datahmifitb/felis-taxonomy-image-classification/data


## Getting Data
The project utilizes the <a href="https://www.kaggle.com/datasets/datahmifitb/felis-taxonomy-image-classification/data">Felis Taxonomy Image Classification</a> dataset from Kaggle, containing 519 JPG images of seven Felis species:

* Domestic cat (F. catus)
* European wildcat (F. silvestris)
* Jungle cat (F. chaus)
* African wildcat (F. lybica)
* Black-footed cat (F. nigripes)
* Sand cat (F. margarita)
* Chinese mountain cat (F. bieti)

![alt text](https://github.com/polaternez/felis-image-classification/blob/master/reports/figures/train_images.png "Train images")


## Data Preprocessing
- **Data Split:** The dataset was divided into 80% for training and 20% for testing to ensure the model learns well and is tested fairly..
- **Image Preprocessing:**
  - Resizing: All images were resized to 224x224 pixels to keep them consistent.
- **Image Data Augmentation:** To make the model more adaptable and prevent it from overfitting, these techniques were applied:
  - Image Rotation
  - Image Translation
  - Image Flipping
  - Contrast Adjustment

## Model Building 
The model is based on a fine-tuned EfficientNetB0 pre-trained model with the following architecture:

![alt text](https://github.com/polaternez/felis-image-classification/blob/master/reports/figures/model.png "Convolutional Neural Network(CNN)")


## Model Evaluation 
The model's performance is measured using categorical cross-entropy and optimized using the ADAM algorithm. The results are as follows:

![alt text](https://github.com/polaternez/felis-image-classification/blob/master/reports/figures/model_evaluation.png "Model Performances")


## Productionization 
A user-friendly API is developed using Flask. The API receives image inputs and returns the predicted Felis species.

![alt text](https://github.com/polaternez/felis-image-classification/blob/master/reports/figures/flask-api.png "Felis Image Classification API")







