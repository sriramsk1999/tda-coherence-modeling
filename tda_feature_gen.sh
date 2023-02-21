python feature_gen.py --cuda 1 --data_name clinton_train --input_dir GCDC_Dataset/ --output_dir gcdc_tda_features --batch_size 15
python feature_gen.py --cuda 1 --data_name clinton_test --input_dir GCDC_Dataset/ --output_dir gcdc_tda_features --batch_size 15
python feature_gen.py --cuda 1 --data_name yahoo_train --input_dir GCDC_Dataset/ --output_dir gcdc_tda_features --batch_size 15
python feature_gen.py --cuda 1 --data_name yahoo_test --input_dir GCDC_Dataset/ --output_dir gcdc_tda_features --batch_size 15
python feature_gen.py --cuda 1 --data_name enron_train --input_dir GCDC_Dataset/ --output_dir gcdc_tda_features --batch_size 15
python feature_gen.py --cuda 1 --data_name enron_test --input_dir GCDC_Dataset/ --output_dir gcdc_tda_features --batch_size 15
python feature_gen.py --cuda 1 --data_name yelp_train --input_dir GCDC_Dataset/ --output_dir gcdc_tda_features --batch_size 15
python feature_gen.py --cuda 1 --data_name yelp_test --input_dir GCDC_Dataset/ --output_dir gcdc_tda_features --batch_size 15

