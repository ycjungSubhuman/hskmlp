{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}

module Neural where

  import Numeric.LinearAlgebra
  import Tool

  -- Fully connected neural network described in ADT
  data Network where
    -- Start layer : data layer. consists of just one data vector.
    StartLayer :: Vector Double -> Network
    -- Hidden layer : consists of three elements
    -- 1. network subset that comes before this layer
    -- 2. weight matrix of size ((prev layer output dimension), (curr hidden layer dimension)).
    -- 3. current forward values
    HiddenLayer :: Network -> Matrix Double -> Vector Double -> Network
    -- End layer : softmax layer. consists of three elements
    -- 1. network subset that comes before this layer
    -- 2. weight vector of size ((prev layer output dimension)).
    -- 3. current softmax value
    EndLayer :: Network -> Vector Double -> Vector Double -> Network

  softmax :: Vector Double -> Vector Double
  softmax v = exp v / (size v) |> [normalizer,normalizer..]
    where
      normalizer = (exp v) <.> 1

  jaSoftmax :: Vector Double -> Matrix Double

  sigmoid x = 1.0 / (1 + exp(-x))

  jaSigmoid x = (sigmoid x) * (1 - sigmoid x)

  crossEntropy :: Vector Double -> Vector Double -> Double
  crossEntropy t y = -(sumElements $ (t * (log y)) + ((1 - t) * log (1 - y)))

  jaCrossEntropy :: Vector Double -> Double -> Vector Double

  -- Initialize network with zero weights and zero values
  -- The first argument is dimension of hidden layers. 0th element represents dimension of the last hidden layer
  -- The second argument is dimension of input layer.
  -- length of hiddenDims should be larger than or equal to 1
  initNetwork :: [Int] -> Int -> Network
  initNetwork hiddenDims inputDim =
    EndLayer
      (initInnerNetwork hiddenDims inputDim)
      (zeros $ head hiddenDims)
      (zeros $ head hiddenDims)
      where
        initInnerNetwork hiddenDims inputDim =
          if 1 == length hiddenDims then
            HiddenLayer
              (StartLayer (zeros inputDim))
              (zeromat inputDim (head hiddenDims))
              (zeros (head hiddenDims))
          else
            HiddenLayer
              (initInnerNetwork (tail hiddenDims) inputDim)
              (zeromat (head (tail hiddenDims)) (head hiddenDims))
              (zeros (last hiddenDims))

  -- Get a vector value of the last layer of the network
  lastLayerOf :: Network -> Vector Double
  lastLayerOf network = case network of
    StartLayer d -> d
    HiddenLayer _ _ value -> value
    EndLayer _ _ value -> value

  -- Prepare Network with input
  feedData :: Vector Double -> Network -> Network
  feedData input network = case network of
    StartLayer d -> StartLayer input
    HiddenLayer prevLayer weights value -> HiddenLayer (feedData input prevLayer) weights value
    EndLayer prevLayer weight value -> EndLayer (feedData input prevLayer) weight value

  -- Forward data to update values of Network. Leaves weights of layers unchanged.
  -- Use this for prediction
  forward :: Network -> Network
  forward network = case network of
    StartLayer d -> StartLayer d
    HiddenLayer prevLayer weights _ ->
      HiddenLayer prevResult weights (sigmoid (weights #> lastLayerOf prevResult))
        where prevResult = forward prevLayer
    EndLayer prevLayer weight _ ->
      EndLayer prevResult weight (softmax $ weight * lastLayerOf prevResult)
        where prevResult = forward prevLayer

  -- Backpropagate errors to update weights of Network. Leaves values of layers unchanged.
  -- Use this for training
  backward :: Network -> Network
  backward network = case network of
    StartLayer d -> StartLayer d
    HiddenLayer prevLayer _ value -> HiddenLayer prevResult () value
      where prevResult = backward prevLayer
    EndLayer prevLayer weight value ->
