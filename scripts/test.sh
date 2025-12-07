
python varying.py -m \
  experiment=test \
  data=Custom_dataset_deep \
  relevance=deep \
  logging_policy_ranker=linear \
  relevance_tower=deep \
  bias_tower=multi_embedding \
  policy_strength=1 \
  policy_temperature=0 \
  random_state=2021 \
  use_baidu=True \
  baidu_subset=train_Baidu_ULTRA_part1_media_type_position.npz \
  $@
