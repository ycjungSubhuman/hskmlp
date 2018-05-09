module Mnist.Load where

  import Neural
  import Numeric.LinearAlgebra
  import Debug.Trace
  import qualified Data.ByteString.Lazy as B
  import Data.Binary.Get
  import Data.Int
  import Control.Monad

  -- MNIST binary data parsers
  getLabelHeader :: Get Int
  getLabelHeader = do
    _ <- getWord32be
    num <- getInt32be
    return $ fromIntegral num

  getLabel :: Get (Vector Double)
  getLabel = do
    label <- getInt8
    return $ (toOneHot . fromIntegral) label
      where
        toOneHot :: Int -> Vector Double
        toOneHot v = vector $ (\i -> if i==v then 1.0 else 0.0) `map` [0..9]

  genVector :: Get (Vector Double) -> Get [Vector Double]
  genVector getOne = do
    empty <- isEmpty
    if empty then return []
    else do h <- getOne
            tl <- genVector getOne
            return (h:tl)

  getImageHeader :: Get (Int, Int, Int)
  getImageHeader = do
    _ <- getWord32be
    num <- getInt32be
    w <- getInt32be
    h <- getInt32be
    return (fromIntegral num, fromIntegral w, fromIntegral h)

  getImage :: Get (Vector Double)
  getImage = do
    colors <- replicateM (28*28) getWord8
    return $ vector ((toRealColor . fromIntegral) `map` colors)

  toRealColor :: Int -> Double
  toRealColor v = (fromIntegral v) / 255

  -- Parse MNIST Dataset into a list of examples (pair of a list of images and a list of labels)
  loadExamples :: IO [Example]
  loadExamples = do
    rawImage1 <- B.readFile "data/train-images.idx3-ubyte"
    rawImage2 <- B.readFile "data/t10k-images.idx3-ubyte"
    rawLabel1 <- B.readFile "data/train-labels.idx1-ubyte"
    rawLabel2 <- B.readFile "data/t10k-labels.idx1-ubyte"
    let labels1 = runGet (getLabelHeader *> (genVector getLabel)) rawLabel1
    let labels2 = runGet (getLabelHeader *> (genVector getLabel)) rawLabel2
    let images1 = runGet (getImageHeader *> (genVector getImage)) rawImage1
    let images2 = runGet (getImageHeader *> (genVector getImage)) rawImage2
    return $ zip (images1++images2) (labels1++labels2)
