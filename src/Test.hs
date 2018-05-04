module Test where

  import Neural(Example)
  import Numeric.LinearAlgebra
  import Tool

  -- Prints Test result to stdout
  runTest :: [Example] -> (Vector Double -> Vector Double) -> IO ()
  runTest examples predictFunction =
    do
      putStrLn "Confusion Matrix : "
      disp 6 $ getConfusion predictions gts
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

