{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}

module Neural(
             Example,
             initNetwork,
             train,
             predict
             ) where

  import Numeric.LinearAlgebra
  import Tool

  type Example = (Vector Double, Vector Double)

  -- Fully connected neural network described in ADT
  data Network where
    -- Start layer : data layer. consists of just one data vector. (homogeneous)
    StartLayer :: Vector Double -> Network
    -- Hidden layer : consists of three elements
    -- 1. network subset that comes before this layer
    -- 2. weight matrix of size ((curr hidden layer dimension), (prev layer output dimension+1)).
    -- 3. current forward values (homogeneous)
    HiddenLayer :: Network -> Matrix Double -> Vector Double -> Network
    -- End layer : softmax layer. consists of three elements
    -- 1. network subset that comes before this layer
    -- 2. current softmax value (homogeneous)
    EndLayer :: Network -> Vector Double -> Network

  softmax :: Vector Double -> Vector Double
  softmax v = exp v / (size v) |> [normalizer,normalizer..]
    where
      normalizer = (exp v) <.> 1

  crossEntropy :: Vector Double -> Vector Double -> Double
  crossEntropy t y = -(sumElements $ (t * (log y)) + ((1 - t) * log (1 - y)))

  sigmoid x = 1.0 / (1 + exp(-x))

  -- Derivation computations

  jaSoftmax :: Vector Double -> Matrix Double
  jaSoftmax smValue = -(smValue `outer` smValue) + (diag smValue)

  {-
   - jaNodeHidden is a matrix of derivatives
   -
   - dy1/dx1 dy1/dx2 ... dy1/dxr
   -          ...
   - dyk/dx1 dyk/dx2 ... dyk/dxr
   -
   - where yi = sigmoid (sum (weight * xi)), dimension of x is r, dimension of y is k.
   -}
  jaNodeHidden :: Matrix Double -> Vector Double -> Matrix Double
  jaNodeHidden weights y = weights * (repeatCols (cols weights) jaSigmoid)
    where
      jaSigmoid = (sigmoid y) * (1 - sigmoid y)

  jaWeightHidden :: Vector Double -> Vector Double -> Matrix Double
  jaWeightHidden x y = jaSigmoid `outer` jaLinear
    where
      jaSigmoid = (sigmoid y) * (1 - sigmoid y)
      jaLinear = x

  jaCrossEntropy :: Vector Double -> Vector Double -> Vector Double
  jaCrossEntropy t y = fromList $ (\i -> (t!i / y!i) - ((1 - t!i) / (1 - y!i))) `map` [0..(size y)-1]

  -- Initialize network with zero weights and zero values
  -- The first argument is dimension of hidden layers. 0th element represents dimension of the last hidden layer
  -- The second argument is dimension of input layer.
  -- length of hiddenDims should be larger than or equal to 1
  initNetwork :: [Int] -> Int -> Network
  initNetwork hiddenDims inputDim =
    EndLayer
      (initInnerNetwork hiddenDims inputDim)
      (zeros $ head hiddenDims)
      where
        initInnerNetwork hiddenDims inputDim =
          if 1 == length hiddenDims then
            HiddenLayer
              (StartLayer (homo $ zeros inputDim))
              (zeromat (head hiddenDims) (inputDim+1)) -- col+1 for bias
              (homo $ zeros (head hiddenDims)) -- append 1 for bias
          else
            HiddenLayer
              (initInnerNetwork (tail hiddenDims) inputDim)
              (zeromat (head hiddenDims) (head (tail hiddenDims) + 1)) -- col+1 for bias
              (homo $ zeros (last hiddenDims)) -- append 1 for bias

  -- Get a vector value of the last layer of the network
  lastLayerOf :: Network -> Vector Double
  lastLayerOf network = case network of
    StartLayer d -> d
    HiddenLayer _ _ value -> value
    EndLayer _ value -> value

  -- Train FC neural network with examples
  train :: Double -> [Example] -> Network -> Network
  train learningRate examples network = case examples of
    [] -> network
    (input, gt):tl -> train learningRate tl improvedNetwork
      where improvedNetwork = ((backward learningRate gt) . forward . (feedData input)) network

  predict :: Vector Double -> Network -> Vector Double
  predict input network = (lastLayerOf . forward . (feedData input)) network

  -- Prepare Network with input
  feedData :: Vector Double -> Network -> Network
  feedData input network = case network of
    StartLayer d -> StartLayer $ homo input
    HiddenLayer prevLayer weights value -> HiddenLayer (feedData input prevLayer) weights value
    EndLayer prevLayer value -> EndLayer (feedData input prevLayer) value

  -- Forward data to update values of Network. Leaves weights of layers unchanged.
  -- Use this for prediction
  forward :: Network -> Network
  forward network = case network of
    StartLayer d -> StartLayer d
    HiddenLayer prevLayer weights _ ->
      HiddenLayer prevResult weights (homo (sigmoid (weights #> lastLayerOf prevResult)))
        where prevResult = forward prevLayer
    EndLayer prevLayer _ ->
      EndLayer prevResult (softmax $ invhomo lastVector) -- remove the last element since it is meaningless
        where prevResult = forward prevLayer
              lastVector = lastLayerOf prevResult

  -- Backpropagate errors to update weights of Network. Leaves values of layers unchanged
  -- Uses stochastic gradient descent
  backward :: Double -> Vector Double -> Network -> Network
  backward learningRate gt (EndLayer prevLayer value) =
    EndLayer (innerBackward prevLayer lastDiff) value
      where
        {-
         - lastDiff is a vector of partial derivatives
         -
         - [dL/dY1 ... dL/dYk 1]
         -
         - where last softmax layer is k-dimension.
         - The hidden layer right before this softmax layer has k+1 dimension since it has '1' at the end.
         -}
        lastDiff = homo $ (jaSoftmax value) #> (jaCrossEntropy value gt)

        innerBackward :: Network -> Vector Double -> Network
        innerBackward network nextDiffs = case network of
          StartLayer d -> StartLayer d
          HiddenLayer prevLayer weights values -> HiddenLayer (innerBackward prevLayer currDiff) newWeights values
            where
              {-
               - currDiff is a vector of partial derivatives
               -
               - [dL/dX1 ... dL/dXr]
               -
               - where X is the output of previous layer
               -}
              currDiff = (tr (jaNodeHidden weights values)) #> nextDiffs
              newWeights = weights - ((\w -> w*learningRate) `cmap` weightDiff)
                where
                  {-
                   - weightDiff is a matrix filled with partial derivatives
                   -
                   -  dL/dW11  dL/dW12 ...  dL/dW1r
                   -               ...
                   -  dL/dWk1  dL/dWk2 ...  dL/dWkr
                   -
                   - where this hidden layer have 'k' nodes and previous layer has 'r' nodes
                   -}
                  weightDiff = pjacobian * (repeatCols (cols pjacobian) nextDiffs)
                    where
                      {-
                       - pjacobian is a matrix filled with partial derivatives (p is for pseudo)
                       -
                       -  dY1/dW11  dY1/dW12 ...  dY1/dW1r
                       -               ...
                       -  dYk/dWk1  dYk/dWk2 ...  dYk/dWkr
                       -
                       - where this hidden layer have 'k' nodes and previous layer has 'r' nodes
                       -}
                      pjacobian = let x = lastLayerOf prevLayer in jaWeightHidden x values

