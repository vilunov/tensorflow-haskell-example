{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}

module Lib (
  main,
  Model(..),
) where

import Control.Monad (forM_)
import System.Random (randomIO)

import Data.ByteString (ByteString)
import Data.Vector.Sized (Vector, replicateM)
import GHC.TypeNats (Nat, KnownNat)

import TensorFlow.DepTyped hiding (Tensor)
import qualified TensorFlow.Core as TF (Tensor, Build, Scalar(Scalar), run, run_, runSession, build, unScalar)
import qualified TensorFlow.Tensor as TF (toTensor)
import qualified TensorFlow.Minimize as TF (adam)
import qualified TensorFlow.Variable as TF (Variable, readValue)
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.Logging as TF
import qualified TensorFlow.DepTyped.Output as TFD (unControlNode)
import qualified TensorFlow.DepTyped.Tensor as TFD (Tensor(Tensor))
import Data.ProtoLens (decodeMessageOrDie)

type Tensor (a :: [Nat]) = TFD.Tensor a '[] TF.Build Float
type Parameter (a :: [Nat]) = Variable a Float

data Model (n :: Nat) = Model {
  train ::
    TensorData "X" '[n] Float ->
    TensorData "Y" '[n] Float ->
    Session ByteString,
  infer ::
    TensorData "X" '[n] Float ->
    Session (Vector n Float),
  slopeValue :: Tensor '[1],
  interceptValue :: Tensor '[1]
}

model :: TF.Build (Model 100)
model = do
  -- Prediction
  (inputs :: Placeholder "X" '[100] Float) <- placeholder
  (slope :: Parameter '[1]) <- initializedVariable 0
  (intercept :: Parameter '[1]) <- initializedVariable 0
  let predictions = (inputs `mul` readValue slope) `add` readValue intercept
  predict <- render predictions

  let
    infer' x = do
      let feeds = feed inputs x :~~ NilFeedList
      runWithFeeds feeds predict

  -- Training
  (outputs :: Placeholder "Y" '[100] Float) <- placeholder
  let
    loss = square (predictions `sub` outputs)
    params = [unVariable slope, unVariable intercept]
  trainStep <- minimizeWith TF.adam loss params

  TF.scalarSummary "slope" $ TF.readValue $ unVariable slope
  TF.scalarSummary "intercept" $ TF.readValue $ unVariable intercept
  st :: TF.SummaryTensor <- TF.mergeAllSummaries

  let
    train' x y = do
      let feeds = feed inputs x :~~ feed outputs y :~~ NilFeedList
      () <- runWithFeeds feeds trainStep
      TF.Scalar sum <- TF.run st
      pure sum

  pure Model {
    train = train',
    infer = infer',
    slopeValue = readValue slope,
    interceptValue = readValue intercept
  }

generate ::
  forall (n :: Nat).
  (KnownNat n) =>
  IO (Vector n Float, Vector n Float)
generate = do
  xData <- replicateM randomIO
  let yData = fmap (\x -> x * 3 + 8) xData
  pure (xData, yData)


type N = 100

fit ::
  Vector N Float ->
  Vector N Float ->
  TF.EventWriter ->
  IO (Float, Float)
fit xData yData ew = TF.runSession $ do
  TF.logGraph ew model
  -- Create tensors with data for x and y.
  let
    x = encodeTensorData xData :: TensorData "X" '[N] Float
    y = encodeTensorData yData :: TensorData "Y" '[N] Float
  m <- TF.build model
  let
    TFD.Tensor s = slopeValue m
    TFD.Tensor i = interceptValue m
  forM_ [1..20000] $ \step -> do
    summary <- train m x y
    TF.logSummary ew step $ decodeMessageOrDie summary
  TF.Scalar s' <- TF.run s
  TF.Scalar i' <- TF.run i
  pure (s', i')


main :: IO ()
main = do
  -- Generate data where `y = x*3 + 8`.
  (xData :: Vector 100 Float, yData) <- generate
  -- Fit linear regression model.
  (w, b) <- TF.withEventWriter "logs" $ fit xData yData
  print w
  print b
