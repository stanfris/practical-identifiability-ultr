
module purge
module load 2023
module load Anaconda3/2023.07-2  # You may not need Anaconda if UV is installed separately
source .venv/bin/activate


python main.py -m \
  experiment=test_varying \
  data=Custom_dataset \
  relevance=linear \
  logging_policy_ranker=linear \
  relevance_tower=linear \
  policy_strength=1 \
  policy_temperature=0 \
  random_state=2021 \
  $@
