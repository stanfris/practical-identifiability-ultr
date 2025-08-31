#!/bin/bash

python main.py -m \
  experiment=3-model-misfit-omitted-features \
  data=mslr30k \
  features=0-99 \
  relevance=deep \
  logging_policy_ranker=deep \
  relevance_tower=deep \
  policy_strength=1,0,-1 \
  policy_temperature=0,0.333,0.666,1.0 \
  random_state=2021,2022,2023 \
  use_propensity_weighting=False,True \
  $@
