# House Price Estimator

## Table of contents
* [Overview](#overview)
* [Goals](#goals)
* [Getting Started](#getting-started)

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

3. You are now ready to run the demo or make a new prediction

```
$ python3 demo.py
$ python3 makeprediction.py <list type,location,sqm,level,number of beds,number of baths,year of construction>  <algorithm (optional)>
```