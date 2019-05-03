# LASERWordEmbedder
This project aims to extract multilingual deep contextualized word embeddings from the LSTM encoder presented in [Facebook LASER](https://arxiv.org/abs/1812.10464)

##TODO: adapt readme to project.

## Setup
1. setup anaconda environment by running from the home folder:
```conda env create -f environment.yml```
2. Run setup as described in ReadME in LASER folder.

Important: the LASER folder has to be added to the PATH variable (in the current session) in order for the code to work. See point 2 for more info.

## Interface
The main functionalities of this repo are train.py, eval.py and infer.py, run them as:
``` python script_to_run.py --flag1 val1 --flagn valn ```
The model_type flag described below must always be either "BOW", "SimpleLSTM", "SimpleBiLSTM" or "PooledBiLSTM".

### Train options
```
model_type  | STR | Name of encoder to be used
model_name | STR | Name of model files to be saved
checkpoint_path | STR | Path to save the trained model
embedding_path | STR | Path to load embeddings from
learning_rate | FLOAT | Initial learning rate for training
batch_size | INT | Number of sequences per batch
embedding_dimension | INT | Number of dimensions to encode sentence onto
weight_decay | FLOAT | Weight regularization parameter
verbose |  BOOL | Whether to print logging during execution
```
### Eval options
```
model_type  | STR | Name of encoder to be used
checkpoint_path | STR | Path to load the trained model from
word_mapping_path | STR |  Path to load the word mapping from
embedding_path | STR | Path to load embeddings from
learning_rate | FLOAT | Initial learning rate for training
batch_size | INT | Number of sequences per batch
embedding_dimension | INT | Number of dimensions to encode sentence onto
eval_setname | STR | Name of the set to evaluate on (must be either train, dev or test)
```
### Infer options
```
model_type  | STR | Name of encoder to be used
checkpoint_path | STR | Path to load the trained model from
word_mapping_path | STR |  Path to load the word mapping from
embedding_path | STR | Path to load embeddings from
learning_rate | FLOAT | Initial learning rate for training
batch_size | INT | Number of sequences per batch
embedding_dimension | INT | Number of dimensions to encode sentence onto
in_path | STR | Path to load the sentences from
out_path | STR | Path to load the encodings to
```
## Notebooks
In order to reproduce the results of the trained models on the [FaceBook SentEval](https://github.com/facebookresearch/SentEval) dataset, senteval.ipynb can be used.
One can also experiment with the trained models by using Demo.ipynb. Output can be examined at sample level and error analysis with visualizations can be performed. Finally, the notebook can be used to visualize the embeddings created by the model by using TSNE dimensionality reduction.




