module Tool where
  import Numeric.LinearAlgebra
  import Data.List

  zeros n = n |> [0,0..]

  zeromat n m = (n><m) [0,0..]

  -- append 1 at the end
  homo :: Vector Double -> Vector Double
  homo v = vjoin [v, vector [1]]

  -- remove 1 at the end
  invhomo :: Vector Double -> Vector Double
  invhomo v = subVector 0 (size v - 1) v

  repeatCols :: Int -> Vector Double -> Matrix Double
  repeatCols n v = repmat (asColumn v) 1 n

  stepLabel :: Vector Double -> Int
  stepLabel v = argmax (toList v)

  labelEquals :: Vector Double -> Vector Double -> Bool
  labelEquals a b = (argmax $ toList a) == (argmax $ toList b)

  argmax l = case elemIndex (maximum l) l of Just result -> result

  -- split a list into two lists
  split l = [take halfLength l, drop halfLength l]
    where halfLength = length l `quot` 2
