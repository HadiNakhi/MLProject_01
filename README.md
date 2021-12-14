# MLProject_01
This project impelemented for midterm of the Machine Learning #Zoomcamp #Alexey Grigorev
## Context

## Dataset

## Feature Description


### bank client data:

## Repo Structure

The following files are included in the repo:

```
heart-failure-prediction
├── Dockerfile <- Docker file with specifications of the docker container
├── Pipfile <- File with names and versions of packages installed in the virtual environment
├── Pipfile.lock <- Json file that contains versions of packages, and dependencies required for each package
├── Procfile <- Procfile for Cloud deployment
├── README.md <- Getting started guide of the project
├── .csv <- Dataset
├── model.bin <- Exported trained model
├── notebook.ipynb <- Jupyter notebook with all codes
├── predict.py <- Model prediction API script for local deployment
├── preduct_test.py <- Model prediction API script for testing
├── requirements.txt <- Package dependency management file
└── train.py <- Final model training script
```

## Create a Virtual Environment

Clone the repo:

```
git clone <repo>
cd MLProject_01 
```

For the project, **virtualenv** is used. To install virtualenv:

```
pip install virtualenv
```

To create a virtual environment:

```
virtualenv venv
```

If it doesn't work then try:

```
python -m virtualenv venv
```

## Activate the Virtual Environment:

For Windows:

```
.\venv\Scripts\activate
```

For Linux and MacOS:

```
source venv/bin/activate
```

## Install Dependencies

Install the dependencies:

```
pip install -r requirements.txt
```

## Build Docker Image

To build a Docker image:

```
docker build -t  .
```

TO run the image as a container:

```
docker run --rm -it -p 9696:9696 :latest
```

To test the prediction API running in docker, run `_test.py` locally.

## Run the Jupyter Notebook

Run Jupiter notebook using the following command assuming we are inside the project directory:

```
jupyter notebook
```

## Run the Model Locally

The final model training codes are exported in this file. To train the model:

```
python train.py
``` 

For local deployment, start up the Flask server for prediction API:

```
python predict.py
```

Or use a WSGI server, Waitress to run:

```
waitress-serve --listen=0.0.0.0:9696 predict:app
```

It will run the server on localhost using port 9696.

Finally, send a request to the prediction API `http://localhost:9696/predict` and get the response:

```
python predict_test.py
```

## Run the Model in Cloud 

The model is deployed on **Heroku ** and can be accessed using:

```
https://bank-marketing-system.herokuapp.com/predict
```

The API takes a JSON array of records as input and returns a response JSON array.

How to deploy a basic Flask application to Pythonanywhere can be found [here](https://github.com/nindate/ml-zoomcamp-exercises/blob/main/how-to-use-pythonanywhere.md). 
Only upload the `.csv`, `train.py`, and `.py` files inside the app directory.
Then open a terminal and run `train.py` and `predict.py` files. Finally, reload the application.
If everything is okay, then the API should be up and running.

To test the cloud API, again run `_test.py` from locally using the cloud API URL.
