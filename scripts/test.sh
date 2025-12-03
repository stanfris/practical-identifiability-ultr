
python main.py -m \
  experiment=test \
  data=Custom_dataset_deep \
  relevance=linear \
  logging_policy_ranker=linear \
  relevance_tower=linear \
  policy_strength=1 \
  policy_temperature=0 \
  random_state=2021 \
  use_baidu=True \
  $@
