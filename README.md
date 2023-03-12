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

Train/test MLP using generated TDA features:

```py
python predict.py --input_dir gcdc/GCDC_dataset/ --feat_dir gcdc/gcdc_tda_features/  --domain clinton
```

Finetune transformer:
```py
python finetune.py --input_dir wikipedia_data/wikipedia_dataset/ --data_name wikipedia_global --model_type hat --cuda 0 --batch_size 2
```

