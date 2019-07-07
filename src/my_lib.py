import os
from google.cloud import bigquery
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/johnherr/.google_api_key/john_bigquery_key2.json"
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
# use ggplot style
plt.style.use('ggplot')


def get_data (SQL_query, filename):
    '''Performs a SQL query on Google Bigquery Databases

    REQUIRES:
        (1) Google account with billing
        (2) Project configured with bigquery api
    IMPORTANT:
        Bigquery queries can search huge amounts of data.  Online GUI provides
        query validation and scan size estimate
    ARGS:
        SQL_query - str:
            e.g.
            "SELECT app_id
            FROM `patents-public-data.uspto_oce_office_actions.office_actions`
            WHERE rejection_101 = "1"
            LIMIT 100"
        filename: filename to save query as: '''

    client = bigquery.Client()
    df = client.query(SQL_query).to_dataframe()  # API request
    df.to_pickle(filename)

def open_saved_pickle(name):
    ''' Opens pickel and applies Regex to exctract claim 1'''

    df = pd.read_pickle(name)
    df = get_first_claim(df, 'filed_claims')
    df = get_first_claim(df, 'granted_claims')
    df = df.dropna()
    return df


def get_first_claim(df, col):
    ''' Uses regex to select only claim 1 from a string
    ARGS: pandas DataFrame
    RETURNS: pandas DataFrame    '''

    df[col] = df[col].apply(lambda x: regex_claim(x))
    return df

def regex_claim(string):
    ''' helper function for selecting first claim'''

    pattern = re.compile(r'1\s?\.\s*A([^2])*')
    try:
        result = pattern.search(string)[0]
    except:
        result = None
    return result

def split_data(df, test_size=.20):
    '''Split pandas DataFrame into train and test sets
    '''
    y = np.append(np.ones(df.shape[0]),np.zeros(df.shape[0])) #rejected and then granted claims
    X = np.append(df['filed_claims'].values, df['granted_claims'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    return X_train, X_test, y_train, y_test

def word_tokenizer(doc, ngram_max=3):
    '''Custom tokenizer breakiing claims into term frequency matricies
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    claim_tokens = tokenizer.tokenize(doc.lower())
    lemmatizer = WordNetLemmatizer()
    claim_tokens = [lemmatizer.lemmatize(w) for w in claim_tokens]
    # stop_words = set(['1.', '1 .', 'a', 'the', 'to'])
    # claim_tokens = [w for w in claim_tokens if w not in stop_words]
    output = []
    for i in range(ngram_max):
        a = list(ngrams(claim_tokens, i+1))
        output.append(a)
    return [item for items in output for item in items]


def get_term_matrix(X_train, X_test, max_features=140, min_df=1, ngrams=1):
    '''Get both TF matrix and TF-IDF matrix
    ARGS: X train/test dataset
    RETURN: TF and TF-IDF matrix for both X_train and X_test, names is list of features
    '''
    cv = CountVectorizer(tokenizer=word_tokenizer, max_features=max_features, min_df=min_df)
    tfidf_transformer = TfidfTransformer(norm='l1')

    X_tr_tf = cv.fit_transform(X_train) #returns document term matrix
    X_tr_tfidf = tfidf_transformer.fit_transform(X_tr_tf)

    X_te_tf = cv.transform(X_test)
    X_te_tfidf = tfidf_transformer.transform(X_te_tf)

    names = cv.get_feature_names()

    return X_tr_tf, X_tr_tfidf, X_te_tf, X_te_tfidf, names


def scree_plot(ax, pca, n_components_to_plot=8, title=None, save = False):
    """Make a scree plot showing the variance explained (i.e. variance
    of the projections) for the principal components in a fit sklearn
    PCA object.

    Parameters
    ----------
    ax: matplotlib.axis object
    The axis to make the scree plot on.

    pca: sklearn.decomposition.PCA object.
    A fit PCA object.

    n_components_to_plot: int
    The number of principal components to display in the scree plot.

    title: str
    A title for the scree plot.
    """

    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)

    for i in range(num_components):
        ax.annotate(r"{:2.2f}%".format(vals[i]),
            (ind[i]+0.2, vals[i]+0.005),
            va="bottom",
            ha="center",
            fontsize=12)

    ax.set_xticklabels(ind, fontsize=12)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=16)

    if save:
        plt.savefig('images/{}.jpg'.format(title), dpi= 300)

def plot_two_componets(ax, X, y, title=None, save=False):
    """Plot an embedding of the mnist dataset onto a plane.

    Parameters
    ----------
    ax: matplotlib.axis object
    The axis to make the scree plot on.

    X: numpy.array, shape (n, 2)
    A two dimensional array containing the coordinates of the embedding.

    y: numpy.array
    The labels of the datapoints.  Should be digits.

    title: str
    A title for the plot.
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax.axis('off')
    ax.patch.set_visible(False)
    ax.scatter(X[:, 0], X[:, 1], c = y, alpha=.5)
    ax.set_xticks([]),
    ax.set_yticks([])
    ax.set_ylim([-0.1,1.1])
    ax.set_xlim([-0.1,1.1])
    if title is not None:
        ax.set_title(title, fontsize=16)
    if save:
        plt.savefig('images/{}.png'.format(title), dpi= 300)

def variance_explained(ax, pca, title='Explained variance', save = False):
    total_variance = np.sum(pca.explained_variance_)
    cum_variance = np.cumsum(pca.explained_variance_)
    prop_var_expl = cum_variance/total_variance
    ax.plot(prop_var_expl, color = 'black', linewidth=2, label=title)
    ax.axhline(0.9, label='90% goal', linestyle='--', linewidth=1)
    ax.set_ylabel('proportion of explained variance')
    ax.set_xlabel('number of principal components')
    ax.legend()
    if save:
        plt.savefig('img/{}.jpg'.format(title), dpi= 300)

def get_top_vals(model, names, num=10):
    '''
    Returns the top words
    '''
    indx = np.argsort(model.coef_)[0][::-1] #largest to smallest
    names = np.array(names)
    ordered_values = names[indx 
    coef = (model.coef_)[0][indx]
    return ordered_values[:num],coef[:num]

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=18, save=False):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    sns.set(font_scale=2)
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save:
        plt.savefig('images/confusion_matrix.png', dpi= 300, bbox_inches="tight")
    return fig
