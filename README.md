# Language Models Repository

Current repository contains experiments on language modeling for text classification. 

## Steps to be done
- [x] Prepare train/test/val
- [ ] Get TM results
- [x] Implement QRNN
- [x] Implement BiQRNN
- [ ] Implement simple LSTM
- [ ] Implement BiLSTM
- [ ] Add SVM and XgBoost tools
- [x] Milestone: initial classification results (19 oct.)
- [x] Implement VAE  

## TODO's
- [x] Prepare new test/train/ver with extracted entities
- [x] Find out why loss becomes NaN in QRNN (too high learning rate)
- [ ] Preprocess embedding data to zero mean and unit variance  
- [x] Check on models_dir/model_1542027692.h5 weights (w2v embedding)  
=======================================================================
- [ ] Prepare test/train/verif sets with less than 10 tokens
- [x] Check processed comments (reprocess), and other sets
- [ ] Prepare model with fasttext embeddings (1 - without preprocessing)   
- [ ] Prepare model with fasttext embeddings (2 - lemmatized)   
- [ ] Reduce the dictionary and substitute rare words with oov (?)
- [ ] Change percentage of positive examples in training set (?)
- [ ] Tune model with hyperopt
- [ ] Check model_1542229255 on comments with more than 50 tokens
- [ ] Prepare report on language model
- [ ] Rewrite to pipeline
=======================================================================  
- [x] Try simple bilstm  
- [ ] Find out  what's happening inside of neural network  (in progress)
- [ ] Try ELMo embeddings  (in progress)
- [ ] Add context  
=======================================================================  
More data 
- [x] Divide on chunks  
- [x] Add TripAdvisor proocessed comments to train/test
- [x] Introduce new train/test/ver v5 

 ## Datasets
 
* '/mnt/shdstorage/tmp/classif_tmp/X_train.csv'
* '/mnt/shdstorage/tmp/classif_tmp/X_test.csv'
* '/mnt/shdstorage/tmp/classif_tmp/y_train.csv'
* '/mnt/shdstorage/tmp/classif_tmp/y_test.csv'
 
## Literature

1. Stephen Merity, Nitish Shirish Keskar, and Richard Socher. Regularizing and optimizing LSTM language models. CoRR, abs/1708.02182,  2017.   URL http://arxiv.org/abs/1708.02182.
2. J. Howard and S. Ruder.  Universal language model fine-tuning for text classification. Association for Computational Linguistics (ACL), 2018.
3. Bradbury, J., Merity, S., Xiong, C., and Socher, R. Quasi-Recurrent Neural Networks. arXiv preprint arXiv:1611.01576, 2016.
4. Kutuzov A., Kuzmenko E. (2017) WebVectors: A Toolkit for Building Web Interfaces for Vector Semantic Models. In: Ignatov D. et al. (eds) Analysis of Images, Social Networks and Texts. AIST 2016. Communications in Computer and Information Science, vol 661. Springer, Cham  
5. A. Odena and I. Goodfellow, “Tensorfuzz: Debugging neural networks with coverage-guided fuzzing,” arXiv preprint arXiv:1807.10875, 2018.  
6. Karpathy, A.; Johnson, J.; and Li, F.-F. 2015. Visualizing and understanding recurrent networks. arXiv preprint.