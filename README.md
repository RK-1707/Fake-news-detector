# Fake-news-detector
This repository contains a deep learning model of a fake news detector which when given a input tells if a given news is real or fake with an accuracy of 93%.

The dataset (29MBs) can be found here https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view. 

The program uses a Tfidfvectoriser library from sklearn. Often times, when building a model with the goal of understanding text, you’ll see all of stop words being removed.The function of the library is to convert a collection of raw documents into a matrix of TF-IDF features.

Term Frequency (TF): The number of times a word appears in a document divded by the total number of words in the document. Every document has its own term frequency.

Inverse Data Frequency (IDF): The log of the number of documents divided by the number of documents that contain the word 'w'. Inverse data frequency determines the weight of rare words across all documents in the corpus.
