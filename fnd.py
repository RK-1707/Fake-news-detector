import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Loading data
os.chdir("C:\lenovo\python_datasets")
data_csv= pd.read_csv('news.csv')
labels= data_csv.label

#Split the dataset
x_train,x_test, y_train,y_test= train_test_split(data_csv['text'], labels, test_size=0.2, random_state=7)

#Initialize a TfidfVectorizer
tfidf_vectorizer= TfidfVectorizer(stop_words='english', max_df=0.7)
#Fit and transform train set, transform test set
tfidf_train= tfidf_vectorizer.fit_transform(x_train) 
tfidf_test= tfidf_vectorizer.transform(x_test)

#Initialize a PassiveAggressiveClassifier
pac= PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#Predict on the test set and calculate accuracy
y_pred= pac.predict(tfidf_test)
score= accuracy_score(y_test,y_pred)
print("Accuracy: " +str(score*100)+ "%")

#Build confusion matrix
print(confusion_matrix(y_test,y_pred, labels=['FAKE','REAL']))