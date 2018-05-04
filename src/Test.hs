module Test where

  import Numeric.LinearAlgebra
  import Data.List

  import Neural
  import Tool

  runSuite :: Double -> [Int] -> ([Example], [Example], [Example]) -> IO ()
  runSuite learningRate hiddenDims (training, validation, test) = do
    putStrLn "Validation Phrase"
    runTest validation (predict network)
    putStrLn "Test Phrase"
    runTest test (predict network)
      where
        network = train learningRate training initialNetwork
          where initialNetwork = initNetwork hiddenDims (size $ (fst.head) training)

  -- Prints Test result to stdout
  runTest :: [Example] -> (Vector Double -> Vector Double) -> IO ()
  runTest examples predictFunction =
    do
      putStrLn "Confusion Matrix : "
      disp 6 $ confusion
      putStr "Precision : "
      putStrLn $ show (getPrecision confusion)
        where
          confusion = getConfusion predictions gts
            where
              predictions = (predictFunction . fst) `map` examples
              gts = snd `map` examples

  -- Row for prediction, col for ground truth
  getConfusion :: [Vector Double] -> [Vector Double] -> Matrix Double
  getConfusion pred gts =
    accum (zeromat width height) (+) ((\p -> (p, 1)) `map` pts)
      where
        width = size (head pred)
        height = width
        pts = zip (stepLabel `map` pred) (stepLabel `map` gts)

  -- Computes precision from a confusion matrix
  getPrecision :: Matrix Double -> Double
  getPrecision confusion = (sumElements $ takeDiag confusion) / (sumElements confusion)

  -- Splits examples into test/validation/test sets
  -- Use half of dataset as training set.
  -- Use the other half as validation/training, equally divided.
  splitDataSet :: [Example] -> [([Example], [Example], [Example])]
  splitDataSet wholeData = do
      [l, r] <- halfPermutations wholeData
      [r_l, r_r] <- halfPermutations r
      return (l, r_l, r_r)
        where
          halfPermutations l = permutations $ split l


