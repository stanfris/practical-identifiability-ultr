#!/bin/bash

python main.py -m \
  experiment=2-model-fit-embedding \
  data=mslr30k \
  use_cross_validation=True \
  relevance=original \
  logging_policy_ranker=expert \
  relevance_tower=embedding \
  policy_strength=1,0,-1 \
  policy_temperature=0,0.333,0.666,1.0 \
  random_state=2021,2022,2023 \
  $@
