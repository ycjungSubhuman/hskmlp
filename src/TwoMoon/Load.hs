module TwoMoon.Load where

  import Neural
  import Numeric.LinearAlgebra

  loadExamples :: IO [Example]
  loadExamples = do
    twomoon <- loadMatrix "data/two_moon.txt"
    let x = vector `map` toLists (twomoon ¿ [0, 1])
    let gt = mkOneHot `map` toList (flatten (twomoon ¿ [2]))
    return $ zip x gt
      where
        mkOneHot :: Double -> Vector Double
        mkOneHot x = if x == 0.0 then vector [1.0, 0.0] else vector [0.0, 1.0]
