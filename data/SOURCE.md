Source: UCI Machine Learning Repository, Steel Plates Faults, dataset ID 198.
Official page: https://archive.ics.uci.edu/dataset/198/steel+plates+faults
DOI: https://doi.org/10.24432/C5J88N
Citation: Buscema, M., Terzi, S., & Tastle, W. (2010). Steel Plates Faults.
License: Creative Commons Attribution 4.0 (CC BY 4.0).

Run python train_models.py on BITS Virtual Lab to download and generate
`steel_plates_prepared.csv`, `test_data.csv`, the models and model metrics.
If the download is blocked, download the original ZIP from UCI manually and use
`python train_models.py --data-path '/path/to/steel+plates+faults.zip'`.

Original UCI file Faults.NNA has 27 numeric features and seven one-hot targets.
We validate exactly one target per row, then replace those flags by Fault_Type.
This conversion cannot introduce any new target information.
