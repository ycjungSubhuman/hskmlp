module Tool where
  import Numeric.LinearAlgebra

  zeros n = n |> [0,0..]

  zeromat n m = (n><m) [0,0..]

