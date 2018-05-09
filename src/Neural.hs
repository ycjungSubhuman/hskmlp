{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}

module Neural(
             Example,
             initNetwork,
             train,
             predict
             ) where

  import Numeric.LinearAlgebra
  import System.Random
  import Debug.Trace

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
    -- 2. current softmax value
    -- 3. current loss of training (not used)
    EndLayer :: Network -> Vector Double -> Double -> Network
    deriving (Show)

  {- Activations and loss -}

  softmax :: Vector Double -> Vector Double
  softmax v = normalizer `seq` ((\val -> val / normalizer) `cmap` (exp v))
    where normalizer = sumElements (exp v)

  crossEntropy :: Vector Double -> Vector Double -> Double
  crossEntropy t y = -(sumElements $ (t * (log y)))

  activation x = tanh x

  {- Derivation computations -}

  {-
   - jaNodeHidden is a matrix of derivatives
   -
   - dy1/dx1 dy1/dx2 ... dy1/dxr
   -          ...
   - dyk/dx1 dyk/dx2 ... dyk/dxr
   -
   - where yi = activation (sum (weight * xj)), dimension of x is r, dimension of y is k.
   -}
  jaNodeHidden :: Matrix Double -> Vector Double -> Matrix Double
  jaNodeHidden weights y = weights * (repeatCols (cols weights) jaActivation)
    where
      jaActivation = 1 - (y^2)

  {-
   - jaWeightHidden is a matrix of derivatives
   -
   - dy1/dw11 dy1/dw12 ... dy1/dw1r
   -          ...
   - dyk/dwk1 dyk/dwk2 ... dyk/dwkr
   -
   - where yi = activation (sum (wij * xj)), dimension of x is r, dimension of y is k.
   -}
  jaWeightHidden :: Vector Double -> Vector Double -> Matrix Double
  jaWeightHidden x y = jaActivation `outer` jaLinear
    where
      jaActivation = 1 - (y^2)
      jaLinear = x

  -- Derivatives of softmax-crossentropy loss function
  jaLoss t y = y - t

  {- Network handling -}

  -- Initialize network with random weights and zero values
  -- The first argument is dimension of hidden layers.
  --   0th element represents dimension of the last hidden layer.
  --   0th element should be equal to the number of classes
  -- The second argument is dimension of input layer.
  -- length of hiddenDims should be larger than or equal to 1
  initNetwork :: [Int] -> Int -> IO Network
  initNetwork hiddenDims inputDim = do
    inner <- initInnerNetwork hiddenDims inputDim
    return $ EndLayer
      inner
      (zeros $ head hiddenDims)
      0.0
      where
        initInnerNetwork :: [Int] -> Int -> IO Network
        initInnerNetwork hiddenDims inputDim =
          if 1 == length hiddenDims then do
            randWeight <- randmat (-0.5, 0.5) (head hiddenDims) (inputDim+1) -- col+1 for bias
            return $ HiddenLayer
              (StartLayer (homo $ ones inputDim))
              randWeight
              (homo $ zeros (head hiddenDims)) -- append 1 for bias
          else do
            randWeight <- randmat (-0.5, 0.5) (head hiddenDims) (head (tail hiddenDims) + 1) -- col+1 for bias
            prevLayer <- (initInnerNetwork (tail hiddenDims) inputDim)
            return $ HiddenLayer
              prevLayer
              randWeight
              (homo $ zeros (last hiddenDims)) -- append 1 for bias

  -- Get a vector value of the last layer of the network
  lastLayerOf :: Network -> Vector Double
  lastLayerOf network = case network of
    StartLayer d -> d
    HiddenLayer _ _ value -> value
    EndLayer _ value _ -> value

  -- Train FC neural network 'epoch' times using given examples
  train :: Int -> Double -> [Example] -> Network -> Network
  train epoch learningRate examples network = examples `seq` ((iterate (trainEpoch learningRate examples) network)!!epoch)

  -- Train FC neural network with examples
  trainEpoch :: Double -> [Example] -> Network -> Network
  trainEpoch learningRate examples network = case examples of
    [] -> network
    (input, gt):tl -> trainEpoch learningRate tl improvedNetwork
      where improvedNetwork = ((backward learningRate gt) . forward . (feedData input)) network

  -- Predict softmax values of network
  predict :: Network -> Vector Double -> Vector Double
  predict network input = (lastLayerOf . forward . (feedData input)) network

  -- Prepare Network with input
  feedData :: Vector Double -> Network -> Network
  feedData input network = input `seq` case network of
    StartLayer d -> StartLayer $ homo input
    HiddenLayer prevLayer weights value -> HiddenLayer (feedData input prevLayer) weights value
    EndLayer prevLayer value loss -> EndLayer (feedData input prevLayer) value loss

  -- Forward data to update values of Network. Leaves weights of layers unchanged.
  -- Use this for prediction
  forward :: Network -> Network
  forward network = case network of
    StartLayer d -> StartLayer d
    HiddenLayer prevLayer weights _ ->
      HiddenLayer prevResult weights (homo (activation (weights #> lastLayerOf prevResult)))
        where prevResult = forward prevLayer
    EndLayer prevLayer _ loss ->
      EndLayer prevResult (softmax $ invhomo lastVector) loss -- remove the last element since it is meaningless
        where prevResult = forward prevLayer
              lastVector = lastLayerOf prevResult

  -- Backpropagate errors to update weights of Network. Leaves values of layers unchanged
  -- Uses stochastic gradient descent
  backward :: Double -> Vector Double -> Network -> Network
  backward learningRate gt (EndLayer prevLayer value _) =
    gt `seq` EndLayer (innerBackward prevLayer lastDiff) value loss
      where
        {-
         - lastDiff is a vector of partial derivatives
         -
         - [dL/dY1 ... dL/dYk]
         -
         - where last softmax layer is k-dimension.
         -}
        lastDiff = jaLoss gt value
        loss = crossEntropy gt value

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
              currDiff = (invhomo . flatten) $ ((jaNodeHidden weights (invhomo values)) * (repeatCols (cols weights) nextDiffs)) ? [0]
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
                      pjacobian = let x = lastLayerOf prevLayer in jaWeightHidden x (invhomo values)

