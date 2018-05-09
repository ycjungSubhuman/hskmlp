module TwoMoon where

  import Numeric.LinearAlgebra
  import System.Environment

  import Neural
  import Test
  import TwoMoon.Load

  main :: IO ()
  main = do
    twomoon <- loadExamples
    learningRate <- (getArgs >>= (\args -> (return.read) $ head args))
    epoch <- (getArgs >>= (\args -> (return.read) $ args!!1))
    let dataSet = splitDataSet twomoon
    runSuite epoch learningRate networkSetting dataSet

  -- two hidden layer with 20, 10 nodes each
  networkSetting = [2, 4]

