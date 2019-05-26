# wildfire-detection-from-satellite-images-ml

### In this project, we detect forest wildfire from given satellite images. I have used CNN, with a training dataset of 2000 images. 

## Demo:
<img src="pics/demo.gif"/>

## Model training
Refer to the research.ipynb jupyter notebook to know the steps taken for model development and algorithm.

## Installing and running this app:
1. Requirements:
Use pip install to download following packages
  - Tensorflow and Keras
  - Flask
  - WSGI server (see the error message when running Flask app, and install all specified packages)
  
 2. running the app:
  - run command: python app.py in the project folder
  - Once the server starts, open browser, the app runs on http://127.0.0.1:5000/
  - "test satellite images" folder contains some satellite images that you can upload to check the working of machine learning application
  
