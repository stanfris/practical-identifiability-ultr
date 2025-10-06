import os
import random
import shutil
import numpy as np


def create_custom_dataset(initial_path, file, 
                          num_groups=1, docs_per_group=10, 
                          D=100, s_group=0.5, s_doc=0.5, 
                          random_seed=42):
    """
    Generate a synthetic LTR-style dataset using a hierarchical Gaussian model
    and balanced quantile-based relevance labels (1–5).
    """
    rng = np.random.default_rng(random_seed)
    os.makedirs(initial_path, exist_ok=True)

    path = os.path.join(initial_path, file)

    # global feature vector
    global_feature_vector = rng.normal(0, 1, D)
    # independently sampled global weight vector
    global_weight_vector = rng.normal(0, 1, D)

    all_scores = []
    all_data = []

    for group_id in range(num_groups):
        group_sampled_features = rng.normal(0, 1, D)
        group_feature_vector = (1 - s_group) * global_feature_vector + s_group * group_sampled_features

        for j in range(docs_per_group):
            individually_sampled_feature_vector = rng.normal(0, 1, D)
            individual_feature_vector = (1 - s_doc) * group_feature_vector + s_doc * individually_sampled_feature_vector

            score = float(np.dot(global_weight_vector, individual_feature_vector))
            all_scores.append(score)
            all_data.append((j//10, individual_feature_vector))

    # Compute 5-level relevance bins using quantiles
    thresholds = np.percentile(all_scores, [20, 40, 60, 80])

    def score_to_label(s):
        if s <= thresholds[0]: return 1
        elif s <= thresholds[1]: return 2
        elif s <= thresholds[2]: return 3
        elif s <= thresholds[3]: return 4
        else: return 5

    # Write file in RankLib/LibSVM format
    with open(path, 'w') as f:
        for (score, (qid, features)) in zip(all_scores, all_data):
            label = score_to_label(score)
            feature_str = ' '.join(f"{k+1}:{features[k]:.4f}" for k in range(D))
            f.write(f"{label} qid:{qid} {feature_str}\n")

    print(f"✅ Balanced dataset written to {path} (quantile-based relevance)")

def write_custom_dataset(initial_path, file, data, num_groups=1, 
                         docs_per_group=10, D=100, s_group=0.5, s_doc=0.5, 
                        random_seed=42):
    test_file = "test.txt"
    validation_file = "vali.txt"
    train_file = "train.txt"

    custom_path = f"num_groups{num_groups}_docs{docs_per_group}_D{D}_sgroup{s_group}_sdoc{s_doc}"

    test_delete = f"../ltr_datasets/cache/custom_dataset_deep-{custom_path}-1-test.pckl"
    validation_delete = f"../ltr_datasets/cache/custom_dataset_deep-{custom_path}-1-val.pckl"
    train_delete = f"../ltr_datasets/cache/custom_dataset_deep-{custom_path}-1-train.pckl"

    # Remove cached pickle files
    for writefile, deletefile in [(test_file, test_delete), (validation_file, validation_delete), (train_file, train_delete)]:
        if os.path.exists(deletefile):
            print(f"Removing {deletefile} from cache")
            os.remove(deletefile)
        create_custom_dataset(initial_path, writefile, 
                              num_groups=num_groups, docs_per_group=docs_per_group, 
                              D=D, s_group=s_group, s_doc=s_doc)

    # Zip the dataset
    output_path = '../ltr_datasets/download'
    output_filename = f"Custom_dataset_deep-{custom_path}"
    dir_path = initial_path
    os.makedirs(output_path, exist_ok=True)
    shutil.make_archive(os.path.join(output_path, output_filename), 'zip', dir_path)
    print("📦 Dataset zipped and ready!")

if __name__ == "__main__":
    # Paths and setup
    initial_path = '../ltr_datasets/dataset/Custom_dataset_deep/Fold1'

    test_file = "test.txt"
    validation_file = "vali.txt"
    train_file = "train.txt"

    test_delete = "../ltr_datasets/cache/custom_dataset_deep-1-test.pckl"
    validation_delete = "../ltr_datasets/cache/custom_dataset_deep-1-val.pckl"
    train_delete = "../ltr_datasets/cache/custom_dataset_deep-1-train.pckl"

    # Model parameters
    num_groups = 1
    docs_per_group = 100
    D = 100
    s_group = 0.4
    s_doc = 0.3

    # Remove cached pickle files
    for writefile, deletefile in [(test_file, test_delete), (validation_file, validation_delete), (train_file, train_delete)]:
        if os.path.exists(deletefile):
            print(f"Removing {deletefile} from cache")
            os.remove(deletefile)
        create_custom_dataset(initial_path, writefile, 
                              num_groups=num_groups, docs_per_group=docs_per_group, 
                              D=D, s_group=s_group, s_doc=s_doc)

    # Zip the dataset
    output_path = '../ltr_datasets/download'
    output_filename = "Custom_dataset_deep"
    dir_path = initial_path
    os.makedirs(output_path, exist_ok=True)
    shutil.make_archive(os.path.join(output_path, output_filename), 'zip', dir_path)
    print("📦 Dataset zipped and ready!")