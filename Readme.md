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