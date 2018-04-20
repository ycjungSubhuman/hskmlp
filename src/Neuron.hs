{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}

module Neural where

  import qualified Numeric.LinearAlgebra.Static as L

  data Layer dimInput dimOutput where
    Single :: L.R a -> L.R b -> Layer a b
