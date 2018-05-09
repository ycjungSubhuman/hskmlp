module Tool where
  import Numeric.LinearAlgebra
  import Data.List
  import System.Random

  {- Convenience functions -}

  -- zero vector of length n
  zeros n = n |> [0,0..]

  -- one vector of length n
  ones n = n |> [1,1..]

  -- random vector of length n
  rands range n = do
    randarr <- (getStdGen >>= (return.randomRs range))
    return $ n |> randarr

  -- zero matrix of (n, m) shape
  zeromat n m = (n><m) [0,0..]

  -- one matrix of (n, m) shape
  onemat n m = (n><m) [1,1..]

  -- random matrix of (n, m) shape
  randmat range n m = do
    randarr <- (getStdGen >>= (return.randomRs range))
    return $ (n><m) randarr

  -- append 1 at the end of a vector
  homo :: Vector Double -> Vector Double
  homo v = vjoin [v, vector [1]]

  -- remove 1 at the end of a vector
  invhomo :: Vector Double -> Vector Double
  invhomo v = subVector 0 (size v - 1) v

  -- Make a matrix by repeating a column vector
  repeatCols :: Int -> Vector Double -> Matrix Double
  repeatCols n v = repmat (asColumn v) 1 n

  -- Given one-hot label, convert it to a index
  stepLabel :: Vector Double -> Int
  stepLabel v = argmax (toList v)

  -- Check if two one-hot labels are equal
  labelEquals :: Vector Double -> Vector Double -> Bool
  labelEquals a b = (argmax $ toList a) == (argmax $ toList b)

  -- Find the index of maximum element in a list
  argmax l = case elemIndex (maximum l) l of Just result -> result

  -- Split a list into half
  split l = [take halfLength l, drop halfLength l]
    where halfLength = length l `quot` 2
