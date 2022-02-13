# House Price Estimator

## Table of contents
* [Overview](#overview)
* [Goals](#goals)
* [Getting Started](#getting-started)
* [Run Demo](#run-demo)
* [Get Today's Data](#get-today's-data)
* [Create Today's Best Model](#create-today's-best-model)
* [Make A Prediction](#make-a-prediction)

## Overview

This project contains a complete house price estimator for the city of Piraeus. It came to fruition in the context of a machine learning excercise and should be treated as such.

## Goals

The goal of this project is to experiment with and compare different regression models and to create an API which makes an estimation of the value of a house located in Piraeus.

## Getting Started

To run the estimator localy you have to follow these steps:

1. Clone repo

```
$ git clone git@github.com:DimitrisPatiniotis/house_valuation.git
```

2. Create a virtual environment and install all requirements listed in requirements.txt

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Run Demo

To execute default demo run:

```
$ python3 demo.py default
```

To execute demo with logarithmic scaling run:

```
$ python3 demo.py scaling log
```

To execute demo with standard scaling run:

```
$ python3 demo.py scaling standard
```

To execute k-NN experiment (examining the relationship between the number of neighbors and model performance) run:

```
$ python3 demo.py test knn
```

To execute Bayesian Ridge Regression experiment (examining the relationship between the number of iterations and model performance) run:

```
$ python3 demo.py test brr
```

To execute Support Vector Regression experiment (examining the relationship between C parameter and model performance) run:

```
$ python3 demo.py test svr
```

Finally, to execute Random Forest Regression experiment (examining the relationship between the number of estimators and model performance) run:

```
$ python3 demo.py test rfr
```

## Get Today's Data

To get today's data run:

```
$ python3 Processes/scraper.py 
```

## Create Today's Best Model

To create an up-to-date inference model (a Random Forest Regressor) run:

```
$ cd Processes/
$ python3 createBestForrest.py
```

Note that the model created has a maximum RMSE of 0.085.

## Make A Prediction

Before you make a prediction, make sure you created have have up to date data and an up to date model (see the two steps mentioned above). Also make sure you are in the project's root directory.

To make a new prediction run:

```
$ python3 makeprediction.py <property_type> <specific_location> <square_meters> <level> <number_of_bedrooms> <number_of_bathrooms> <construction_year>
```

Property type and specific location values are not case sensitive, do not need stress-marks and can even be written in latin characters. An example is given bellow:

```
$ python3 makeprediction.py Διαμέρισμα Πασαλιμανι 90 2 2 1 1971
```