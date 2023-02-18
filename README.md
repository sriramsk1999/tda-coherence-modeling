# Coherence Modeling with TDA

## Setup

```py
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

Generating TDA features for a dataset (in GCDC format):

```py
python feature_gen.py --cuda 1 --data_name clinton_train --input_dir GCDC_Dataset/ --output_dir gcdc_tda_features --batch_size 15
```
