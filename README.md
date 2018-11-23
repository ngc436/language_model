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
- [x] Prepare model with fasttext embeddings (1 - without preprocessing)   
- [x] Prepare model with fasttext embeddings (2 - lemmatized)   
- [X] Reduce the dictionary and substitute rare words with oov (?)
- [ ] Change percentage of positive examples in training set (?)
- [ ] Tune model with hyperopt
- [x] Check model_1542229255 on comments with more than 50 tokens
- [ ] Prepare report on language model
- [ ] Rewrite to normal pipeline
- [ ] Add representation in latent space from VAE 
=======================================================================  
- [x] Try simple bilstm  
- [x] Find out  what's happening inside of neural network (LIME)
- [ ] Prepare ELMo embeddings on raw texts (in progress)
- [ ] Add context (?)  
=======================================================================  
More data 
- [x] Divide on chunks  
- [x] Add TripAdvisor proocessed comments to train/test
- [x] Introduce new train/test/ver v5 
Cleaner data  
- [x] Fix dirty data issue (html, phone instead of id)

## List of experiments
- [ ] BIQRNN fasttext/w2v (time, results, loss plot)
- [ ] BILSTM fasttext/w2v (time, results, loss plot)
- [ ] VAE fasttext/w2v (latent space clustering (if possible), create example transition from negative to positive comment)  (time, results, loss plot)
- [ ] Optimization with hyperopt  
- [ ] Experiment with ELMo embeddings (not sure how yet (? put directly to input without )) pretrained/fine-tuned (if possible)  
#### Additional
- [ ] Look at fasttext work in case of unlemmatized input for the best performing models above (prepare this input) 
- [ ] Try hierarchical attention network


## Datasets
 
 v - dataset version  
 
* '/mnt/shdstorage/tmp/classif_tmp/X_train_v.csv'
* '/mnt/shdstorage/tmp/classif_tmp/X_test_v.csv'
* '/mnt/shdstorage/tmp/classif_tmp/y_train_v.csv'
* '/mnt/shdstorage/tmp/classif_tmp/y_test_v.csv'
 
## Literature

1. Stephen Merity, Nitish Shirish Keskar, and Richard Socher. Regularizing and optimizing LSTM language models. CoRR, abs/1708.02182,  2017.   URL http://arxiv.org/abs/1708.02182.
2. J. Howard and S. Ruder.  Universal language model fine-tuning for text classification. Association for Computational Linguistics (ACL), 2018.
3. Bradbury, J., Merity, S., Xiong, C., and Socher, R. Quasi-Recurrent Neural Networks. arXiv preprint arXiv:1611.01576, 2016.
4. Kutuzov A., Kuzmenko E. (2017) WebVectors: A Toolkit for Building Web Interfaces for Vector Semantic Models. In: Ignatov D. et al. (eds) Analysis of Images, Social Networks and Texts. AIST 2016. Communications in Computer and Information Science, vol 661. Springer, Cham  
5. A. Odena and I. Goodfellow, “Tensorfuzz: Debugging neural networks with coverage-guided fuzzing,” arXiv preprint arXiv:1807.10875, 2018.  
6. Karpathy, A.; Johnson, J.; and Li, F.-F. 2015. Visualizing and understanding recurrent networks. arXiv preprint.