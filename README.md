## Towards Interpretable Solar Flare Prediction with Attention-based Deep Neural Networks
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/Python-3.7.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
[![pytorch](https://img.shields.io/badge/PyTorch-1.10.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this work, we developed an attention-based deep learning model as an improvement over the standard CNN pipeline to perform full-disk binary flare predictions for the occurrence of ≥M1.0-class flares within the next 24 hours.

### Architecture of Standard CNN Pipeline based Full-disk Model

![alt text](https://github.com/chetrajpandey/fulldiskAttention/blob/main/readme_resoc/no_attn_arch.png?raw=true)

### Architecture of the Attention-based Full-disk Model
![alt text](https://github.com/chetrajpandey/fulldiskAttention/blob/main/readme_resoc/attn_arch.png?raw=true)

### Data Labeling
We labeled our data with a prediction window of 24 hours. The images are labeled based on the maximum peak X-ray flux values,
converted to NOAA/GOES flare classes observed in the next 24 hours. See figure below:
![alt text](https://github.com/chetrajpandey/fulldiskAttention/blob/main/readme_resoc/data_label.png?raw=true)

---
### Source Code Documentation

##### 1. download_mag:

This folder/package contains one python module, "download_jp2.py". 
There are two functions inside this module.
First Function: "download_from_helioviewer()" downloads jp2 magnetograms from helioveiwer api : [Helioviewer](https://api.helioviewer.org/docs/v2/api/api_groups/jpeg2000.html)
Second Function: "jp2_to_jpg_conversion()" converts jp2s to jpgs for faster computation. If resize=True, pass height and width to resize the magnetograms

##### 2. data_labeling:

Run python labeling.py : Contains functions to generate labels, binarize, filtering files, and creating 4-fold CV dataset.
Reads goes_integrated_flares.csv files from data_source.
Generated labels are stored inside data_labels. 
labeling.py generates labels with multiple columns that we can use for post result analysis. Information about flares locations, and any other flares that occured with in the period of 24 hours.
For simplification:  folder inside data_labels, named simplified_data_labels that contains two columns: the name of the file and actual target that is sufficient to train the model.

##### 3. modeling:

The code is optimized and can only be used with two cuda devices using nn.dataparallel. For single GPU use please make modifications in train.py under model configuration. It Contains all the trained models inside directory trained_models. attention-based inside attention and standard cnn inside no_attention.<br /> <br /> 
(a) attention_model.py: This module contains the architecture of both the model. Passing attention=True activates attention architecture and False activates standard CNN pipeline.<br /> 
(b) blocks.py: This module contains the linearattention block and projector block required by attention_model.py for attention-based model.<br /> 
(c) dataloader.py: This contains custom-defined data loaders for loading FL and NF class for selected augmentations.<br /> 
(d) evaluation.py: This includes functions to convert tensors to sklearn compatible array to compute confusion matrix. Furthermore TSS and HSS skill scores definition.<br /> 
(e) initialize.py: This module contains Kaiming intialization functions for convolution, batchnorm and linear blocks. Default is KaimingUniform.<br /> 
(f) train.py: This module is the main module to train the model. Uses argument parsers for parameters change. This has seven paramters to change the model configuration:<br /> 
(i) --fold: choose from 1 to 4, to run the corresponding fold in 4CV-cross validation; default=1<br /> 
(ii) --epochs: number of epochs; default=30<br /> 
(iii) --batch_size: default=128<br /> 
(iv) --im_size: Size of the input image; default=256<br /> 
(v) --attention: Select architecture: 1 for attention-based use any integer for standard CNN; default=1<br /> 
(vi) --lr: initial learning rate selection; default=0.001<br /> 
(vii) --weight_decay: regularization parameter used by the loss function; default=0.5<br /> 

For Example: <br /> 
To run the first fold with attention for 50 epochs:<br /> 
python train.py --fold=1 --epochs=50 --attention=1<br /> 
To run the second fold with standard CNN for 10 epochs:<br /> 
python train.py --fold=2 --epochs=10 --attention=0

##### 4. result_analysis:


This folder contains 3 jupyter notebooks for evaluating the models.<br /> 

(a) Attention_Predictions.ipynb : This notebook shows our validation skill scores for all attention-based trained models in 4 folds expt. Furthermore contains flares predictions in central and near-limb locations. <br /> 
(b) No_Attention_Predictions.ipynb : This notebook shows our validation skill scores for all standard CNN trained models in 4 folds expt. Furthermore contains flares predictions in central and near-limb locations.<br /> 
(c) Attention_Maps_Visualize.ipynb: Visualizes 3 instances including two True Positives (east and west limb flares) and one False Positive instance using Attention-Estimator-2 of the trained attention-based model in fold 1. Three instances locations are stored inside plots.csv file.

---
