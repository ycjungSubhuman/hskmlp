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
    let dataSets = splitDataSet twomoon
    runSuite learningRate networkSetting `mapM_` dataSets

  -- one hidden layer with 2 nodes
  networkSetting = [2]

