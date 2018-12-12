# Lazydiff (CS 207 Final Project Group 7)
[![Build Status](https://travis-ci.org/CS207-Project-Group-7/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/CS207-Project-Group-7/cs207-FinalProject.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/CS207-Project-Group-7/cs207-FinalProject/badge.svg?branch=master)](https://coveralls.io/github/CS207-Project-Group-7/cs207-FinalProject?branch=master)


<img src="docs/ForwardAccumulationAutomaticDifferentiation.png" align="right" width="400" height="200">

**Lazydiff** is a library for performing automatic differentiation (AD). AD is a set of techniques to numerically evaluate the derivative of a function motivated by deficiencies in classical methods. Numerical approximation is often not accurate enough and can be computationally expensive and symbolic differentiation tends to lead to inefficient code and faces the difficulty of converting a computer program into a single expression. Both classical methods have problems with calculating higher derivatives, where the complexity and errors increase, and are slow when it comes to computing the partial derivatives of a function with respect to many inputs. Automatic differentiation endeavors to solve all of these problems, and has been used in neural networks for back propagation and weight adjustment based on the loss function, and in scientific computing where it is often difficult to analytically compute the derivatives.

## Full Documentation

See [Documentation](docs/Documentation.ipynb) for full documentation and current progress.

## Installing

If you want to install the package only for the current virtual environment, run the following commands to create and activate a new virtual env:

- Move to the desired working directory
- `virtualenv env`
- `source env/bin/activate`

You can run the following to install the dependencies:
- `pip install -r requirements.txt`

Then execute following command to install LazyDiff and its dependencies:
 - `pip install git+https://github.com/CS207-Project-Group-7/cs207-FinalProject.git`

## Contributors 
* Joe Davison
* Raymond Lin
* Zheng Yang
* Matteo Zhang

## License

MIT License

Copyright (c) 2018 CS207-Project-Group-7

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

