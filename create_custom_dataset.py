# 2 million lines, 30 k queries 
import os
import random
import shutil


def create_custom_dataset(initial_path, file, num_repeats=1, num_queries=10):    
    path = os.path.join(initial_path, file)
    with open(path, 'w') as f:
        for _ in range(num_repeats):
            for x in range(100):
                relevance = random.uniform(0, 10)
                feature_string = ''
                for y in range(100):
                    if y == x:
                        feature_string += f" {y}:{1}"
                    else:
                        feature_string += f" {y}:0"
                f.write(f"{relevance} qid:{x//10} {feature_string} \n")

    return

if __name__ == "__main__":
    initial_path = '../ltr_datasets/dataset/Custom_dataset/Fold1'
    test_file = "test.txt"
    validation_file = "vali.txt"
    train_file = "train.txt"
    test_delete = "../ltr_datasets/cache/custom_dataset-1-test.pckl"
    validation_delete = "../ltr_datasets/cache/custom_dataset-1-val.pckl"
    train_delete = "../ltr_datasets/cache/custom_dataset-1-train.pckl"

    num_repeats = 1
    num_queries = 10

    for writefile, deletefile in [(test_file, test_delete), (validation_file, validation_delete), (train_file, train_delete)]:
        if os.path.exists(deletefile):
            print(f"Removing {deletefile} from cache")
            os.remove(deletefile)
        if writefile == train_file:
            create_custom_dataset(initial_path, writefile, num_repeats=num_repeats, num_queries=num_queries)
        else:
            create_custom_dataset(initial_path, writefile, num_repeats=num_repeats, num_queries=num_queries)

    output_path = '../ltr_datasets/download'
    output_filename = "Custom_dataset"
    dir_name = "Fold1"
    output_path = os.path.join(output_path, output_filename)
    dir_path = os.path.join(initial_path, dir_name)
    shutil.make_archive(output_path, 'zip', dir_path)