# 2 million lines, 30 k queries 
import os
import random
import shutil


def create_custom_dataset(initial_path, file, num_lines=100000, num_queries=100):    
    path = os.path.join(initial_path, file)
    with open(path, 'w') as f:
        for i in range(num_queries):
            for x in range(num_lines // num_queries):
                relevance = x//5
                f.write(f"{relevance} qid:{i} 1:{x} \n")

    return

if __name__ == "__main__":
    initial_path = '../ltr_datasets/dataset/Custom_dataset/Fold1'
    test_file = "test.txt"
    validation_file = "vali.txt"
    train_file = "train.txt"
    test_delete = "../ltr_datasets/cache/custom_dataset-1-test.pckl"
    validation_delete = "../ltr_datasets/cache/custom_dataset-1-val.pckl"
    train_delete = "../ltr_datasets/cache/custom_dataset-1-train.pckl"

    num_samples = 1000000
    num_queries = 10000

    for writefile, deletefile in [(test_file, test_delete), (validation_file, validation_delete), (train_file, train_delete)]:
        if os.path.exists(deletefile):
            print(f"Removing {deletefile} from cache")
            os.remove(deletefile)
        create_custom_dataset(initial_path, writefile, num_lines=num_samples, num_queries=num_queries)

    output_path = '../ltr_datasets/download'
    output_filename = "Custom_dataset"
    dir_name = "Fold1"
    output_path = os.path.join(output_path, output_filename)
    dir_path = os.path.join(initial_path, dir_name)
    shutil.make_archive(output_path, 'zip', dir_path)