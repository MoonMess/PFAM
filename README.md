# Protein classification by Mounir Messaoudi

This Repo contains :
- a pdf resuming all the work done
- data analysis (data_analysis.ipynb)
- code for preprocessing (preprocessign.py)
- code for training (train.py)
- code for evaluation (test.py)
- evaluation analysis (evaluation.ipynb) 

Achieve 99% test accuracy trained by ResNet with Adam optimizer, for classifying 250 classes.

## Hardware
-----------
- Windows 10
- Intel(R) Core(TM) i5
- Memory: 16G
- NVIDIA GeForce GTX 1060 * 1 (4617 MB memory)

## Pipeline
-----------
To reproduce the result, there are three steps:
1. Dataset Preparation, run preprocessing.py
2. Training, for each backcbone, run train.py
3. Evaluation, for each backbone, run test.py

## Environment
-----------
### Requirements
```
#Install requirements
pip install -r requirements.txt

#In case of problems
intall tensorflow, Bio and propythia using pip and it should work.

```
Project directory structured
```
+-- assets/
|   +-- family_distribution.png
|   +-- ...
+-- backbone/
|   +-- layers/
|   |   +--  ResidualBlock.py
|   |   +--  ...
|   +-- resnet.py
|   +-- ...
+-- logs/
+-- output/
|   +-- exp1
|   +-- ...
+-- preprocessed_data
|   +-- full/
|   |   +-- ...
|   +-- sample/
|   |   +-- ...
+-- random_split/
|   +-- train
|   +-- dev
|   +-- test
+-- utils/
|   +-- dataloader.py
|   +-- model.py
|   +-- tools.py
+-- preprocessing.py
+-- README.md
+-- requirements.txt
+-- train.py
+-- test.py
+-- pfam Classification - Mounir Messaoudi.pdf
+-- evaluation.ipynb
+--data_analysis.ipynb
```

## Dataset preparation
---------------

### Download Dataset
If no folder random_split, download and put the folder random_plit inside this repo.
["Kaggle: Pfam seed random split"](https://console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam/random_split)

### Extract random_split.zip to dataset directory.
After downloading and pre-process, the data directory is structured as:
```
+-- dataset/
|   +-- train
|   +-- |   +-- data-00000-of-00080
|   +-- |   +-- ...
|   +-- dev
|   +-- |   +-- data-00000-of-00010
|   +-- |   +-- ...
|   +-- test
|   +-- |   +-- data-00000-of-00010
|   +-- |   +-- ...
```
## Dataset Analysis
---------------
```
Look at the notebook data_analysis.ipynb
```
## Training
---------------
### Usage
usage : preprocessing.py

```
It will preprocess the data from random_split/, and save the preprocessed data in preprocessed_data/.

## Training
---------------
### Usage

```
usage: train.py [--output_version OUTPUT_VERSION] [--data_dir DATA_DIR]
                [--backbone BACKBONE] (REQUIRE) 
                [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                [--seed SEED]
```
value for backbone : resnet, mobilnetv2, light_mobilnetv2, LSTM
The rest of the parameters are optional
### Example Usage

Train resnet

```python
python train.py --output_version resnet1 --num_epochs 5 --batch_size 256 --backbone resnet 
```

Train LSTM

```python
python train.py --output_version LSTM1 --num_epochs 5 --batch_size 256 --backbone LSTM
```

Visualize

```
tensorboard --logdir "./logs"
```

## Training Random Forest  
---------------
### Usage

```
open RF_Training.ipynb and run it all
```


## Evaluate (Test) 
---------------
### Usage

```
usage: test.py [--model_dir DATA_DIR] (REQUIRE)
               [--OUTPUT_dir TRAIN_DIR] [--output_version OUTPUT_VERSION]
```

### Example Usage

test trained model

```python
python test.py --model_dir output/20240107-094325
```

## Result (with adam optimizer)
---------------

|       |Test Acc|  
|:-----:|:----:| 
| resnet | 0.9999  | 
| LSTM   | 0.9947  | 
| mobilenetV2 light| 0.9998  | 
| mobilenetV2 | 0.9997 | 
| Random Forest | 0.02

## Reference
---------------
- [Using Deep Learning to Annotate the Protein Universe](https://www.biorxiv.org/content/10.1101/626507v3.full.pdf)
- [Kaggle: Pfam seed random split](https://console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam/random_split)
- [Automatic structure classification of small proteins using random forest](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-364)
- [Efficacy of different protein descriptors in predicting protein functional families ](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-300#Tab1)
- [Protein Sequence Classification](https://www.kaggle.com/code/habibizadkhah/protein-sequence-classification)
- [Deep Learning to Annotate the Protein Universe](https://www.kaggle.com/code/kaledhoshme/deep-learning-to-annotate-the-protein-universe)
- [Using Deep Learning to Annotate the Protein Universe](https://www.biorxiv.org/content/10.1101/626507v2.full.pdf)
- [Deep Dive into Machine Learning Models for Protein Engineering](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00073)
- [Automatic structure classification of small proteins using random forest](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-364)
- [Pytorch gene sequence classification by protCNN](https://github.com/Jackycheng0808/protcnn/blob/main/README.md)
- [A comprehensive framework for advanced protein classification and function prediction using synergistic approaches: Integrating bispectral analysis, machine learning, and deep learning](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0295805#pone.0295805.ref005)