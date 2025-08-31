#!/bin/bash

python main.py -m \
  experiment=2-model-fit-linear \
  data=mslr30k \
  relevance=linear \
  logging_policy_ranker=linear \
  relevance_tower=linear \
  policy_strength=1,0,-1 \
  policy_temperature=0,0.333,0.666,1.0 \
  random_state=2021,2022,2023 \
  $@
