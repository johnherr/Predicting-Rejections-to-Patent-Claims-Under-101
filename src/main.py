import pandas as pd
import numpy as np
import my_lib as lib
from sklearn.naive_bayes import MultinomialNB as SKMultinomialNB
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlit.pyplot as plt


df = lib.open_saved_pickle('data/art_unit_362.pkl')
X_train, X_test, y_train, y_test = lib.split_data(df)
X_tr_tf, X_tr_tfidf, X_te_tf, X_te_tfidf, names = lib.get_term_matrix(X_train,X_test)
mnb = SKMultinomialNB()


# only using TF
mnb.fit(X_tr_tf, y_train)
print('NB Test Accuracy (TF):', mnb.score(X_te_tf, y_test))
sklearn_predictions_ = mnb.predict(X_te_tf)

#using TF-IDF
mnb.fit(X_tr_tfidf, y_train)
print('NB Test Accuracy (TF-IDF):', mnb.score(X_te_tfidf, y_test))
sklearn_predictions = mnb.predict(X_te_tfidf)

#Evaluate NB model
con_matrix = confusion_matrix(y_test, sklearn_predictions)
lib.print_confusion_matrix(con_matrix, ['101 Rejeciton', 'No 101 Rejection'], save=True)
print(accuracy_score(y_test, sklearn_predictions))
print(precision_score(y_test, sklearn_predictions))
print(recall_score(y_test, sklearn_predictions))


n_components = 2
pca = PCA(n_components=n_components) #pca object
X_pca = pca.fit_transform(X_tr_tfidf.toarray())
X_pca.shape
fig3, ax3 = plt.subplots(1,1, figsize=(8, 8))
lib.plot_two_componets(ax3, X_pca, y_train, title="PCA with 2 Components", save=True)
