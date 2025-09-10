cd ..

python create_custom_dataset.py

cd two-tower-confounding

python main.py -m \
  experiment=test_deep \
  data=Custom_dataset \
  relevance=deep \
  logging_policy_ranker=deep \
  relevance_tower=deep \
  policy_strength=1 \
  policy_temperature=0,0.333,0.666,1.0 \
  random_state=2021 \
  $@
