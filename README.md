# MSIGN: A deep learning framework based on multi-scale interaction graph neural networks for predicting binding of synthetic cannabinoids to receptors

## Dataset
The dataset is collected from two publicly available datasets, pdbbind and bindingdb:
- **PDBbind v2020:** [pdbbind](http://www.pdbbind.org.cn/download.php)
- **BindingDB:** [bindingdb]([Binding Database Home](https://www.bindingdb.org/rwd/bind/index.jsp))


## Requirements  
The environment can be installed using the provided YAML file named `./environment.yaml`.


## Process Raw Data
	1.Run python preprocess_data.py
	2.Run python graph_constructor.py


## Model Training
	Run python train.py

## Prediction using a trained model
	Run python predict.py
	We provide an example prediction: Run python predict_example.py







