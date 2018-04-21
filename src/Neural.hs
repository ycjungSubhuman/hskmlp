{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}

module Neural where

  import Numeric.LinearAlgebra.Static
  import GHC.TypeNats

  data Network dimIn dimOut where
    StartLayer :: (KnownNat a) => Network a a
    HiddenLayer :: (KnownNat a, KnownNat b, KnownNat c) => Network a b -> L c b -> Network a c
    EndLayer :: (KnownNat a, KnownNat d) => Network a d -> R d -> Network a d

  softmax :: (KnownNat d) => R d -> R d
  softmax v = exp v / ((exp v) <.> 1)

  sigmoid :: Double -> Double
  sigmoid x = 1.0 / (1 + exp(-x))

  forward :: (KnownNat a, KnownNat b) => R a -> Network a b -> R b
  forward input network = case network of
    StartLayer -> input
    HiddenLayer prevLayer weights -> weights #> forward input prevLayer
    EndLayer prevLayer weight -> softmax $ weight * forward input prevLayer


