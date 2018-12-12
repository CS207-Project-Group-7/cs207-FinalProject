# Lazydiff (CS 207 Final Project Group 7)
[![Build Status](https://travis-ci.org/CS207-Project-Group-7/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/CS207-Project-Group-7/cs207-FinalProject.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/CS207-Project-Group-7/cs207-FinalProject/badge.svg?branch=master)](https://coveralls.io/github/CS207-Project-Group-7/cs207-FinalProject?branch=master)


<img src="docs/ForwardAccumulationAutomaticDifferentiation.png" align="right" width="400" height="200">

**Lazydiff** is a library for performing automatic differentiation (AD). AD is a set of techniques to numerically evaluate the derivative of a function motivated by deficiencies in classical methods. Numerical approximation is often not accurate enough and can be computationally expensive and symbolic differentiation tends to lead to inefficient code and faces the difficulty of converting a computer program into a single expression. Both classical methods have problems with calculating higher derivatives, where the complexity and errors increase, and are slow when it comes to computing the partial derivatives of a function with respect to many inputs. Automatic differentiation endeavors to solve all of these problems, and has been used in neural networks for back propagation and weight adjustment based on the loss function, and in scientific computing where it is often difficult to analytically compute the derivatives.

## Full Documentation

See [Documentation](docs/documentation.ipynb) for full documentation and current progress.

## Installing

The easy way using pip:

- `pip install lazydiff`

You can also execute following command to install LazyDiff and its dependencies:

- `pip install git+https://github.com/CS207-Project-Group-7/cs207-FinalProject.git`

If you are cloning our repo and installing lazydiff that way, you can run the following to install:

- `python3 setup.py install`

and run the following to run our tests:

- `python3 setup.py test`


## Contributors 
* Joe Davison
* Raymond Lin
* Zheng Yang
* Matteo Zhang

