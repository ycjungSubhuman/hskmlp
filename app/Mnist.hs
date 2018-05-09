module Mnist where

  import Numeric.LinearAlgebra
  import System.Environment

  import Neural
  import Test
  import Mnist.Load

  main :: IO ()
  main = do
    mnist <- loadExamples
    learningRate <- (getArgs >>= (\args -> (return.read) $ head args))
    epoch <- (getArgs >>= (\args -> (return.read) $ args!!1))
    let dataSet = splitDataSet mnist
    runSuite epoch learningRate networkSetting dataSet

  -- one hidden layer with 300 nodes
  networkSetting = [10, 300]
