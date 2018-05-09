# hskmlp

Haskell implementation of multi-layer perceptron using stochastic gradient descent.

## Build

1. Install haskell stack (https://docs.haskellstack.org/en/stable/README/)
1. cd into root of this project, where README is located.
1. issue `stack build`


## Run

* `stack exec -- TwoMoon \<learningRate\> \<epochs\>` will run a demo program that runs training/validation/test for two_moon.txt dataset

Empirically 0.03, 1000 gives the best results.

* `stack exec -- Mnist \<learningRate\> \<epochs\>` will run a demo program that runs training/validation/test for MNIST dataset

Empirically 0.03, 20 gives the best results. (This takes about 6 hours on Intel i7-2600)


## Dataset

MNIST dataset in `data/` is from http://yann.lecun.com/exdb/mnist/, and is not a part of this project.


