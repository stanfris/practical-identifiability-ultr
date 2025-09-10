# 2 million lines, 30 k queries 
import os
import json
import random
import shutil


def create_custom_dataset(initial_path, file, num_lines=100000, num_queries=10000):    
    path = os.path.join(initial_path, file)
    with open(path, 'w') as f:
        for _ in range(num_lines):
            query = random.randrange(num_queries)
            relevance = random.randrange(5)
            f.write(f"{relevance} qid:{query} 1:{relevance} 2:{relevance}\n")

    return

if __name__ == "__main__":
    initial_path = '../ltr_datasets/dataset/Custom_dataset'
    num_samples = 200000
    num_queries = 100
    file = "Fold1/train.txt"
    create_custom_dataset(initial_path, file, num_lines=num_samples, num_queries=num_queries)
    file = "Fold1/test.txt"
    create_custom_dataset(initial_path, file, num_lines=num_samples, num_queries=num_queries)
    file = "Fold1/vali.txt"
    create_custom_dataset(initial_path, file, num_lines=num_samples, num_queries=num_queries)

    output_path = '../ltr_datasets/download'
    output_filename = "Custom_dataset"
    dir_name = "Fold1"
    output_path = os.path.join(output_path, output_filename)
    dir_path = os.path.join(initial_path, dir_name)
    shutil.make_archive(output_path, 'zip', dir_path)