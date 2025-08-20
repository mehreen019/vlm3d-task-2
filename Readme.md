## Setup environment

Install conda and run the following command in git bash to create a new environment with the required packages.

```bash
bash setup_env.sh
```

Activate the environment by running:

```bash
conda activate vlm3d_challenge
```

Deactivate the environment by running:

```bash
conda deactivate    
```

## Download CT-RATE dataset

Download the CT-RATE dataset using the `ct_rate_downloader.py` script.

Import your huggingface token before running `ct_rate_downloader.py` script.

```python
import os
os.environ["HF_TOKEN"] = "your_token"
```
Adjust the amount of samples to be downloaded by changing the `--max-storage-gb` argument.
Also enable `--download-volumes` to download the actual CT volumes.

```bash
python ct_rate_downloader.py --max-storage-gb 10 --download-volumes
```

## 2d slice extraction

The 2D slice extraction script is located in `2d_slice_extractor.py`.
It uses the `ct_rate_data` directory as the input and creates the `ct_rate_2d` directory for the output.

```bash
python 2d_slice_extractor.py --data-dir ./ct_rate_data --output-dir ./ct_rate_2d --strategy multi_slice --slices-per-volume 12 
```