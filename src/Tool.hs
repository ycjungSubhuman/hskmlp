module Tool where
  import Numeric.LinearAlgebra

  zeros n = n |> [0,0..]

  zeromat n m = (n><m) [0,0..]

  -- append 1 at the end
  homo :: Vector Double -> Vector Double
  homo v = vjoin [v, vector [1]]

  -- remove 1 at the end
  invhomo :: Vector Double -> Vector Double
  invhomo v = subVector 0 (size v - 1) v

