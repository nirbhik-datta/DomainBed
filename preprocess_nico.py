import os
import argparse
import csv


def rename_environment_dirs(data_dir, high_level_class):
    animal_path = os.path.join(data_dir, "NICO", high_level_class, "images")
    for animal_dir in os.listdir(animal_path):
        animal_dir_path = os.path.join(animal_path, animal_dir)
        environment_dirs = os.listdir(animal_dir_path)
        for environment_dir in environment_dirs:
            if ' ' not in environment_dir:
                continue
            correct_environment_dir = environment_dir.replace(' ', '_')
            environment_dir_path = os.path.join(animal_dir_path, environment_dir)
            correct_environment_dir_path = os.path.join(animal_dir_path, correct_environment_dir)
            os.rename(environment_dir_path, correct_environment_dir_path)


if __name__ == '__main__':
    splits = ['env_train1.csv', 'env_train2.csv', 'env_val.csv', 'env_test.csv']
    root = "C:\\Users\\cthrash\\Downloads\\DomainBed-main\\domainbed\\scripts\\data\\NICO"
    image_root = "C:\\Users\\cthrash\\Downloads\\NICO"
    for split in splits:
        split_csv = os.path.join(root, "mixed_split_corrected", split)

        with open(split_csv) as f:
            reader = csv.reader(f)
            for img_path, category_name, context_name, superclass in reader:
                img_path = img_path.replace('\\', '/')
                full_img_path = os.path.join(image_root, superclass, 'images', img_path)
                if not os.path.isfile(full_img_path):
                    print(full_img_path)


    parser = argparse.ArgumentParser(description='Preprocesses the NICO dataset')
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir

    change_file_names(data_dir, 'vehicle')
    change_file_names(data_dir, 'animal')
    rename_environment_dirs(data_dir, 'vehicle')
    rename_environment_dirs(data_dir, 'animal')


