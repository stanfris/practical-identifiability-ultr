#!/bin/bash

python main.py -m \
  experiment=3-expert-policy \
  data=mslr30k \
  features=all \
  relevance=original \
  logging_policy_ranker=expert \
  relevance_tower=deep \
  policy_strength=-1 \
  policy_temperature=0,0.333,0.666,1.0 \
  random_state=2021,2022,2023 \
  use_propensity_weighting=False,True \
  $@
