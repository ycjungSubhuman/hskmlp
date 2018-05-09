module Test where

  import Numeric.LinearAlgebra
  import Data.List
  import Debug.Trace

  import Neural
  import Tool

  {- Network test codes-}

  -- Train a network and run validation and test
  runSuite :: Int -> Double -> [Int] -> ([Example], [Example], [Example]) -> IO ()
  runSuite epoch learningRate hiddenDims (training, validation, test) = do
    initialNetwork <- initNetwork hiddenDims (size $ (fst.head) training)
    let network = train epoch learningRate training initialNetwork
    putStrLn "Validation Phase"
    runTest validation (predict network)
    putStrLn "Test Phase"
    runTest test (predict network)

  -- Run test with given examples
  runTest :: [Example] -> (Vector Double -> Vector Double) -> IO ()
  runTest examples predictFunction =
    do
      putStrLn "Confusion Matrix : "
      disp 6 confusion
      putStr "Precision : "
      putStrLn $ show (getPrecision confusion)
        where
          confusion = getConfusion predictions gts
            where
              predictions = (predictFunction . fst) `map` examples
              gts = snd `map` examples

  -- Generate confusion matrix
  -- Row for prediction, col for ground truth
  getConfusion :: [Vector Double] -> [Vector Double] -> Matrix Double
  --getConfusion pred gts | trace ("getConfusion " ++ show pred ++ " " ++ show gts) False = undefined
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
  splitDataSet :: [Example] -> ([Example], [Example], [Example])
  splitDataSet wholeData = training `seq` validation `seq` test `seq` (training, validation, test)
    where
      half = length wholeData `quot` 2
      valLen  = 2 * (length wholeData `quot` 10)
      testLen  = 3 * (length wholeData `quot` 10)

      training = take half wholeData
      validation = ((take valLen) . (drop half)) wholeData
      test = ((take testLen) . (drop valLen) . (drop half)) wholeData

