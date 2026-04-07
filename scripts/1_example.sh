#!/bin/bash

python main.py -m \
  experiment=1-example \
  data=mslr30k \
  features=all \
  relevance=original \
  logging_policy_ranker=deep \
  relevance_tower=deep \
  policy_strength=1.0 \
  policy_temperature=0,0.333,0.666,1.0 \
  random_state=2021,2022,2023 \
  use_propensity_weighting=False \
  train_clicks=1000000 \
  val_clicks=1000000 \
  test_clicks=1000000 \
  $@
