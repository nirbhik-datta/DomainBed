## Data preparation
All data used can be downloaded from https://drive.google.com/drive/folders/1ztqNZhn4PPZpaDvhPihipKKUBqYsqPZ5?usp=sharing.
The below describes the process for acquiring each of the datasets.

```
python domainbed/scripts/download.py --data-dir <data output directory>
```

For CelebA: 
1. download from https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ
2. Extract zip file to img_align_celeba
3. create a CelebA directory under your data directory and copy img_align_celeba into it
4. copy the contents of ood_bench_data_files/celeba into your <data dir>/CelebA directory

for NICO:
1. download from https://pan.baidu.com/s/1yS3vbx788FOtpgY9N_vVIQ password is rpds (NOTE: this requires a Chinese phone number)
2. Rename the "Animal" and "Vehicle" directories to "animal" and "vehicle"
3. run ```python nico_preprocessing.py --data_dir <path to your downloaded dataset>```
4. A few of the image file names are malformed. For example, "vehicle/truck/on_beach/83. filename-img-0989-jpg" should be "vehicle/truck/on_beach/83.jpg". Run ```python nico_misnamed_files.py --data_dir <path to downloaded directory``` to find them all.
5. Create a NICO directory under your data directory
6. Add the "animal" and "vehicle" directories 
7. copy the contents of ood_bench_data_files/NICO into your <data dir>/NICO directory