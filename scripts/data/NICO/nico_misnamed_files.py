import os
import argparse
import csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocesses the NICO dataset')
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir

    splits = ['env_train1.csv', 'env_train2.csv', 'env_val.csv', 'env_test.csv']
    splits_path = "./ood_bench_data_files/NICO/mixed_split_corrected"
    for split in splits:
        split_csv = os.path.join(splits_path, split)

        with open(split_csv) as f:
            reader = csv.reader(f)
            for img_path, category_name, context_name, superclass in reader:
                img_path = img_path.replace('\\', '/')
                full_img_path = os.path.join(data_dir, superclass, 'images', img_path)
                if not os.path.isfile(full_img_path):
                    print(full_img_path)
