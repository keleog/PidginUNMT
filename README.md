# Unsupervised Neural Machine Translation from West African Pidgin (Creole) to English

This repository contains the implementation of an Unsupervised NMT model from West African Pidgin (Creole) to English without a single parallel sentence used during training. 

## Dependencies

* Python 3
* [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/)
* [Moses](http://www.statmt.org/moses/) (for evaluation)
* [NLTK](https://www.nltk.org/) (for tokenization, make sure to download punkt)
* [RCSLS Alignment](https://github.com/facebookresearch/fastText/tree/master/alignment) (for generating cross-lingual embeddings)

## Running Unsupervised NMT

### Aligning Word Vectors
Check the Alignment Folder and run the notebook to align and evaluate the Pidgin Word Vectors to the English Word Vectors. 

### Preprocess data
The first thing to do to run the NMT model is to preprocess data. To do so, just follow the instructions in the 
preprocess.ipynb notebook. This notebook helps convert your text file data to serialized data for training.

Note that although this repository supports the use of BPE codes, all work was done with regular word vectors. 

### Train the NMT model
Please note that some parameters must respect a particular format:
- **langs**
    + A list of languages, sorted by language ID.
    + `en,pd` for "English and Pidgin"
- **mono_dataset**
    + A dictionary that maps a language to train, validation and test files.
    + Validation and test files are optional (usually we only need them for training).
    + `en:en_train.pt,en_valid.pt,en_test.pt;pd:pd_train.pt,pd_valid.pt,pd_test.pt`
- **para_dataset**
    + A dictionary that maps a language pair to train, validation and test files.
    + Training file is optional (in unsupervised MT we only use parallel data for evaluation).
    + `en-pd:XX_para_train.pt,XX_para_valid.pt,XX_para_test.pt` to indicate the validation and test paths.
- **mono_directions**
    + A list of languages on which we want to train the denoising auto-encoder.
    + `en,pd` to train the auto-encoder both on English and Pidgin.
- **para_directions**
    + A list of tuples on which we want to train the MT system in a standard supervised way.
    + `en-pd,pd-en` will train the model in both the `en->pd` and `pd->en` directions.
    + Requires to provide the model with parallel data.
- **pivo_directions**
    + A list of triplets on which we want to perform back-translation.
    + `pd-en-pd,en-pd-en` will train the model on the `pd->en->pd` and `en->pd->en` directions.

**Other parameters:**
- `--otf_num_processes 30` indicates that 30 CPU threads will be generating back-translation data on the fly, using the current model parameters
- `--otf_sync_params_every 1000` indicates that models on CPU threads will be synchronized every 1000 training steps
- `--lambda_xe_otfd 1` means that the coefficient associated to the back-translation loss is fixed to a constant of 1
- `--lambda_xe_mono '0:1,100000:0.1,300000:0'` means that the coefficient associated to the denoising auto-encoder loss is initially set to 1, will linearly decrease to 0.1 over the first 100000 steps, then to 0 over the following 200000 steps, and will finally be equal to 0 during the remaining of the experiment (i.e. we train with back-translation only)

Given binarized monolingual training data, parallel evaluation data, and pretrained cross-lingual embeddings, you can train the model using the following command:

```
python main.py --exp_name en_pd_exp --transformer True --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --langs 'en,pd' --n_mono -1 --mono_dataset 'en:en_train.pt,en_valid.pt,en_test.pt;pd:pd_train.pt,pd_valid.pt,pd_test.pt' --para_dataset 'en-pd:,XX_para_valid.pt,XX_para_test.pt' --mono_directions 'en,pd' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions 'en-pd-en,pd-en-pd' --pretrained_emb 'pidg_vect_RCSLS.txt' --pretrained_out True --save_periodic True--lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --params.lambda_dis = '1' --otf_num_processes 30 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 500000 --stopping_criterion bleu_en_pd_valid,10
```

Although some of these parameters are already specified in the [main.py](./main.py) file, you can overwrite them as you deem fit to suit your usecase. 

## Acknlowledgments and References

This repository borrowed extensively from these repositories - https://github.com/facebookresearch/UnsupervisedMT and https://github.com/facebookresearch/fastText/tree/master/alignment

This work builds extensively on the following works:

[1] A. Joulin, P. Bojanowski, T. Mikolov, H. Jegou, E. Grave, Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion, 2018. (https://arxiv.org/abs/1804.07745)

[2] G. Lample, A. Conneau, L. Denoyer, MA. Ranzato, Unsupervised Machine Translation With Monolingual Data Only, 2018a. (https://arxiv.org/abs/1711.00043)

[3] G. Lample, M. Ott, A. Conneau, L. Denoyer, MA. Ranzato, Phrase-Based & Neural Unsupervised Machine Translation, 2018b. (https://arxiv.org/abs/1804.07755)

## License

See the [LICENSE](LICENSE) file for more details.
