{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}

module Neural where

  import Numeric.LinearAlgebra
  import Tool

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

  -- Jacobian computations

  jaSoftmax :: Vector Double -> Matrix Double
  jaSoftmax smValue = fromList $ (\(i,j) ->
    if i==j then smValue!i * (1 - smValue!i) else -(smValue!i * smValue!j)
    ) `map` zip [0..(size x)-1] [0..(size x)-1]

  jaSigmoid x = (sigmoid x) * (1 - sigmoid x)

  jaLinear x = x

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
  backward eta gt (EndLayer prevLayer value) =
    EndLayer (innerBackward $ prevLayer ((jaSoftmax value) #> (jaCrossEntropy value gt))) value
      where
        innerBackward network nextJ = case network of
          StartLayer d -> StartLayer d
          HiddenLayer prevLayer _ values -> HiddenLayer (innerBackward prevLayer jacobian) neweights values
            where
              jacobian = nextJ
              newweights = 
