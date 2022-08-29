#!/usr/bin/env python
# coding: utf-8

# # TEXT CLASSIFICATION

# In this assignment, you will
# 
# 1. Download text (the IMDB reviews dataset).
# 1. Process it; vectorize it.
# 1. Perform parameter selection (for logistic regression) using cross validation on the train split.
# 1. Train a model using the best parameter setting.
# 1. Report the top positive and negative features and discuss your findings.
# 1. Evaluate the model on the test split. Report performance metrics.
# 1. Choose a few random test documents.
# 1. Print the top positive and negative words in those documents.
# 1. Discuss your findings.

# **Name**: Jane Downer <br>
# **CWID**: A20452471 <br>
# **Section**: 02 <br>

# In[1]:


get_ipython().system('cd  /opt/anaconda3/envs/venv_581')
# !pip install matplotlib
# !pip install nltk
# !pip install pandas
# !pip install scikit-learn
# !pip install seaborn
# !pip install tqdm
import copy
from   IPython.display import display_html 
import matplotlib.pyplot as plt
import nltk
from   nltk import corpus, WordNetLemmatizer
from   nltk.corpus import stopwords
nltk.download(['stopwords','wordnet','omw-1.4'], quiet=True)

import numpy as np
import os
import pandas as pd
from   pandas import DataFrame as df
import random
import re
import seaborn as sns
import sklearn
from   sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from   sklearn.linear_model import LogisticRegression
from   sklearn.model_selection import GridSearchCV, learning_curve

import tqdm
from   tqdm import tqdm


# ## Dataset
# 
# The IMDB review dataset is available via different platforms. We will work with the "raw" text version of it.
# 
# Download the dataset from https://ai.stanford.edu/~amaas/data/sentiment/.
# 
# Unzip it; put it into a folder called ``aclImdb`` (case senstive). The ``aclImdb`` folder should be in the same folder as this notebook. The folder structure should look like this:

# ``aclImdb`` <br>
# -> ``train`` <br>
# ---> ``neg`` <br>
# ---> ``pos`` <br>
# -> ``test`` <br>
# ---> ``neg`` <br> 
# ---> ``pos`` <br>
# 
# You can ignore the other folders.

# The files in the ``train`` folder are train documents and the ``test`` are, well, the test documents. The documents in ``neg`` are negative documents and the documents in ``pos`` are the positive documents.
# 
# Load the documents and their labels.

# In[2]:


dir_ = '/Users/janedowner/Desktop/CS 581/CS 581 - Assignment 4'
os.chdir(dir_)

train = {'File #':[], 'Document': [],'Label':[]}
test  = {'File #':[], 'Document': [],'Label':[]}

def build_dict(dictionary,folder_extension, label):
    folder = os.listdir(dir_ + folder_extension)
    for file in folder:
        with open(dir_ + folder_extension + file) as f:
            dictionary['File #']  .append(file[:-4])
            dictionary['Document'].append(''.join(f.readlines()))
            dictionary['Label']   .append(label)

build_dict(train,'/aclImdb/train/pos/',1)
build_dict(train,'/aclImdb/train/neg/',0)
build_dict(test, '/aclImdb/test/pos/', 1)
build_dict(test, '/aclImdb/test/neg/', 0)

train_df = pd.DataFrame(train).sample(frac=1).reset_index(drop=True)
test_df  = pd.DataFrame(test) .sample(frac=1).reset_index(drop=True)


# In[3]:


def compose(functions, input_):
    while len(functions) > 0:
        inner_most = functions[-1]
        input_     = inner_most(input_)
        functions  = functions[:-1]
    output = input_
    return output

sw       = stopwords.words('english')
lower    = lambda doc: ''.join([w.lower() for w in doc])
no_hyph  = lambda doc: re.sub(r'-',' ',doc)
no_punct = lambda doc: ''.join([w for w in doc if w.isalpha() or w == ' '])
tok      = lambda doc: doc.split()
no_sw    = lambda doc: [w for w in doc if w not in sw]
lemm     = lambda doc: [WordNetLemmatizer().lemmatize(w) for w in doc]

prep_row = lambda df,row: ' '.join(compose([lemm,
                                            no_sw,
                                            tok,
                                            no_punct,
                                            no_hyph,
                                            lower],df.iloc[row,1]))

train_df_preprocessed = train_df.copy()
test_df_preprocessed  = test_df .copy()
for row in tqdm(range(len(train_df))):
    train_df_preprocessed.iloc[row,1] = prep_row(train_df_preprocessed,row)
    test_df_preprocessed .iloc[row,1] = prep_row(test_df_preprocessed,row)

train_df_preprocessed.to_csv('train_df - preprocessed', index=False)
test_df_preprocessed .to_csv('test_df - preprocessed',  index=False)


# ## Vectorization
# 
# You must use the scikit-learn package for vectorization and for the classifier: https://scikit-learn.org/stable/index.html
# 
# You have freedom regarding the vectorizer: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text You can use the `CountVectorizer` or the `TfidfVectorizer` with parameters of your choice.

# In[4]:


train_df_preprocessed = pd.read_csv(filepath_or_buffer='train_df - preprocessed', 
                                    index_col         = False)
test_df_preprocessed  = pd.read_csv(filepath_or_buffer='test_df - preprocessed',  
                                    index_col         = False)
X_train               = train_df_preprocessed.iloc[:,1]
X_test                = test_df_preprocessed .iloc[:,1]
y_train               = train_df_preprocessed.iloc[:,2]
y_test                = test_df_preprocessed .iloc[:,2]

TFIDF                 = TfidfVectorizer()

X_all                 = pd.concat([X_train,X_test],keys=['X_train','X_test'])
BOW_all               = TFIDF.fit_transform(X_all)
train_indices         = range(len(X_train))
test_indices          = range(len(X_train),len(X_train)+len(X_test))
BOW_train             = BOW_all[train_indices,:]
BOW_test              = BOW_all[test_indices, :]
BOW_train_norm        = sklearn.preprocessing.normalize(BOW_train, 
                                                        norm='l2', 
                                                        return_norm=False)
BOW_test_norm         = sklearn.preprocessing.normalize(BOW_test,  
                                                        norm='l2', 
                                                        return_norm=False)


# ## Parameter Selection
# 
# For the classifier, you must use `LogisticRegression`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 
# Use 10-fold cross validation using various parameter settings for `C`. Please use `penalty='l2'`. You can use any solver; the results should not change drastically based on the solver; I do not recommend spending a lot of time on trying different solvers.
# 
# I recommend using https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html. Report the grid search results using a table.

# In[5]:


param_grid = {'penalty': ['l2'],
              'C':       [1.0,5.0,10.0],
              'tol':     [0.0001,0.01,0.1],
              'solver':  ['liblinear']}

LR_model = LogisticRegression()
grid_clf = GridSearchCV(estimator=LR_model,
                        cv=10,
                        param_grid=param_grid,
                        refit=True,
                        return_train_score=True)

grid_clf.fit(BOW_train_norm,y_train)

best_params = df(grid_clf.cv_results_)[['mean_fit_time',
                                        'params',
                                        'mean_train_score']]

best_params.to_csv(dir_ + '/aclImdb/best_params')
pd.set_option("display.max_colwidth",100)
display(best_params)


# #### The following parameters maximize mean training score, while resulting in lower mean fit time than parameters with comparable scores:
# 
#  `C: 10.0`<br>`penalty: l2`<br>`solver: liblinear`<br>`tol: 0.1`

# ## Refit
# 
# Fit a logistic regression on the full training data using the best parameter settings above.

# In[6]:


LR_model = LogisticRegression(C=10.0,
                              penalty='l2',
                              solver='liblinear',
                              tol=0.1)
LR_model.fit(BOW_train_norm,y_train)


# ## Top Model Features
# 
# Present a table of top 20 positive features (the feature names and their coefficients) and another table of top 20 negative features, with respect to the logistic regression weights.

# In[7]:


coefs       = LR_model.coef_[0]
###
idxs_pos20  = np.argpartition(coefs, -20)[-20:]
idxs_neg20  = np.argpartition(coefs,  20)[: 20]
###
feats_pos20 = TFIDF.get_feature_names_out()[idxs_pos20]
feats_neg20 = TFIDF.get_feature_names_out()[idxs_neg20]
###
coefs_pos20 = coefs[idxs_pos20]
coefs_neg20 = coefs[idxs_neg20]

fts_df  = lambda fts,cfs,cols: df({cols[0]:fts,cols[1]:cfs})
sort_df = lambda df,cols,asc: df.sort_values(by=cols[1],
                                             ascending=asc).reset_index(drop=1)

cols_=['Feature','Coefficient']
featsDF_pos = sort_df(fts_df(feats_pos20,coefs_pos20,cols_),cols_,0)
featsDF_neg = sort_df(fts_df(feats_neg20,coefs_neg20,cols_),cols_,1)

dfL_ = featsDF_pos.style.set_table_attributes("style='display:inline'")
dfL_ = dfL_.set_caption('Top 20 Positive Features')
dfR_ = featsDF_neg.style.set_table_attributes("style='display:inline'")
dfR_ = dfR_.set_caption('Top 20 Negative Features')

space = "\xa0" * 10
display_html(dfL_._repr_html_() + space + dfR_._repr_html_(), raw=True)


# ## Evaluation
# 
# Print a classification report on the train and another classification report on the test. Discuss your findings.

# In[8]:


y_pred_train = LR_model.predict(BOW_train_norm)
y_pred_test  = LR_model.predict(BOW_test_norm)

train_report = sklearn.metrics.classification_report(y_train,
                                                     y_pred_train,
                                                     digits=2,
                                                     zero_division='warn')

test_report  = sklearn.metrics.classification_report(y_test,
                                                     y_pred_test,
                                                     digits=2,
                                                     zero_division='warn')

print('Classification Report -- training data:\n{}'.format(train_report))
print('Classification Report -- testing data:\n{}' .format(test_report))


# ***
# ***
# #### The classifier performs better on the training data than on the testing data. This makes sense to a degree, because the classifier has complete information about the training data, but not the testing data. However, it might be an indicator of overfitting, or some other issue. I will discuss this more later.
# 
# #### High accuracy by itself doesn't mean very much without these other scores to back it up, especially when there is class imbalance. However, within each classification report, these scores are similar to eachother and to accuracy. Moreover, there are an equal number of positive and negative samples anyway. Given this scenario, it is appropriate to consider the accuracy scores.
# ***
# ***

# ## Top Features In Each Document
# 
# Pick a few (at least 3) documents at random. For each document:
# 1. Print the document.
# 1. Print its label.
# 1. Present a a table of top 10 positive features (in that document), their weights, and their weights*feature values.
# 1. Present a a table of top 10 negative features (in that document), their weights, and their weights*feature values.
# 
# 
# Note: if your vectorizer is binary, you weigths and weights*feature should be the same.

# In[9]:


'''
I wasn't sure whether the question was asking to choose documents
from the training or testing data (or both). These docuents come
from the testing data.
''' 
keys_    = lambda dict_: list(dict_.keys())
vals_    = lambda dict_: list(dict_.values())
items_   = lambda dict_: list(dict_.items()) 
list_zip = lambda L1,L2: list(zip(L1,L2))

random_indices  = random.sample(range(len(test_df_preprocessed)),5)
random_data     = test_df_preprocessed.iloc[random_indices,:].reset_index(drop=True)
random_docs_raw = test_df.iloc[random_indices,1].tolist()
random_BOW      = BOW_test_norm.toarray()[random_indices,:]


weights_list_pos, weights_list_neg = [],[]
values_list_pos,  values_list_neg  = [],[]
all_feats = TFIDF.get_feature_names_out().tolist()
for i in range(len(random_data)):
    lab = random_data.iloc[i,2]
    feats_idx   = [idx for idx in range(len(random_BOW[i])) if random_BOW[i][idx] != 0]
    feats       = [all_feats[idx] for idx in feats_idx]
    feats_coefs = [coefs[idx] for idx in feats_idx]
    feats_vals  = [random_BOW[i][idx] for idx in feats_idx]

    items = sorted(list_zip(feats,
                            feats_coefs),
                   key=lambda item: item[1],
                   reverse=True)
    
    pos_items, neg_items = items[:10], items[-10:]
    pos_feats, neg_feats = keys_(dict(items))[:10], keys_(dict(items))[-10:]
    
    weights_pos = {k:v for (k,v) in pos_items}
    weights_neg = {k:v for (k,v) in neg_items}

    values     = {k:v for (k,v) in list_zip(feats, feats_vals)}
    values_pos = {k:v for (k,v) in items_(values) if k in pos_feats}
    values_neg = {k:v for (k,v) in items_(values) if k in neg_feats}
    weights_list_pos.append(weights_pos)
    weights_list_neg.append(weights_neg)
    values_list_pos .append(values_pos)
    values_list_neg .append(values_neg)


w_x_v = lambda weights,values,i: [w*v for (w,v) in zip(vals_(weights[i]),
                                                       vals_(values[i]))]
    
for i in range(5):
    sub_file = 'pos' if random_data.Label[i]==1 else 'neg'
    print('~/test/{}/{}:\n'.format(sub_file, random_data.iloc[i,:]["File #"]))
    print(random_docs_raw[i])
    df_pos = df({'Feature':      keys_(weights_list_pos[i]),
                 'Weight':       vals_(weights_list_pos[i]),
                 'Weight*Value': w_x_v(weights_list_pos, values_list_pos,i)})

    df_neg = df({'Feature':      keys_(weights_list_neg[i]),
                 'Weight':       vals_(weights_list_neg[i]),
                 'Weight*Value': w_x_v(weights_list_neg, values_list_neg, i)})

    dfL_ = df_pos.style.set_table_attributes("style='display:inline'")
    dfL_ = dfL_.set_caption('Top 10 Positive Features')
    dfR_ = df_neg.style.set_table_attributes("style='display:inline'")
    dfR_ = dfR_.set_caption('Top 10 Negative Features')
    space = "\xa0" * 10
    display_html(dfL_._repr_html_() + space + dfR_._repr_html_(), raw=True)

    true_label = random_data.Label[i]
    pred_label = y_pred_test[random_indices[i]]
    sums_pos = round(sum(df_pos['Weight*Value']),3)
    sums_neg = round(sum(df_neg['Weight*Value']),3)
    
    print('True label:      {}'.format('POSITIVE' if true_label==1 else 'NEGATIVE'))
    print('Predicted label: {}'.format('POSITIVE' if pred_label==1 else 'NEGATIVE'))
    print('Sum(weights*values) of the features above: {}'.format(sums_pos+sums_neg))
    print('\n\n')


# ## Final Discussion
# 
# Discuss your findings.

# ## 1.
# 
# ### Do you think your model is accurate, based on classification metrics?
# 
# #### When considered simultaneously, precision, recall and F1 can give a more holistic representation of model performance than just an accuracy score alone. As we saw above, they are all between 0.86 and 0.88, which is decent, but not perfect. Another way to represent the model performance is with the ROC (receiver operator characteristic) -- which shows the tradeoff between false positive rates and true positive rates -- and AUC  (area under the ROC curve).

# In[10]:


y_pred_prob = LR_model.predict_proba(BOW_test_norm)
y_          = y_test.tolist()
y_          = np.asarray([[0,1] if y == 1 else [1,0] for y in y_])
fpr, tpr, _ = sklearn.metrics.roc_curve(y_.ravel(), y_pred_prob.ravel())
auc         = round(sklearn.metrics.auc(fpr,tpr),3)

plt.plot(fpr,tpr)
plt.title('ROC-AUC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.annotate("Area Under Curve: {}".format(auc),xy=[0.6,0.05])
plt.show()


# #### A perfect classifier will have an area under the curve value of 1. Here, it's 0.946, which is pretty good. Moreover, the fact that the curve is pulled toward the upper left-hand corner means that at certain threshold levels, the model can achieve high true positive rates without much of an increase in false positive rates. This is what we want.
# #### The confusion matrix can also provide us with information about model accuracy:

# In[11]:


cm     = sklearn.metrics.confusion_matrix(y_test,y_pred_test)
grps   = ['TNR','FPR','FNR','TPR']
pcts   = ["{0:.2%}".format(val) for val in cm.flatten()/np.sum(cm)]
labels = np.asarray(["{}\n{}".format(g,p) for (g,p) in zip(grps,pcts)]).reshape(2,2)

sns.heatmap(cm,annot=labels,fmt='',cmap=sns.color_palette("Blues",as_cmap=True))
sns.set(font_scale = 2)
plt.title('Confusion Matrix')
plt.show()


# #### The confusion matrix shows that false negatives and false positive occur at rates of 6.91% and 5.80%.  In other words, the model guesses incorrectly 6.91+5.80=12.71% of the time. This is acceptable, but could be better.
# #### All of the above classification metrics suggest that the model is acceptably accurate.

# ## 2.
# 
# ### Do you think your model is smart? Answer this question solely based on the classification accuracy.
# #### The model accuracy is 0.87, which is decent. Based on this number alone, the model seems relatively smart, or at the very least, does its job. However, as I've explained above, accuracy by itself doesn't provide a holistic view of model performance.

# ## 3.
# 
# ### Do you think your model is smart? Answer this question solely based on top features in each document.
# #### Some of the top features in each document make sense, and others do not. In my opinion, the top features of these individual documents provide the strongest evidence that the model is *not* smart.
# #### For example, there is no apparent reason why these words should be associated with positive labels:
# 
# `eerie`<br>
# `kitty`<br>
# `ride`<br>
# `see`<br>
# 
# #### ... Or why these words should be associated with negative labels:
# 
# `artsy`<br>
# `croc`<br>
# `grade`<br>
# `three`<br>
# `tree`<br>
# 
# #### It's worth noting that the top features of the training data in its entirety made much more sense. With more data, the strongest trends emerge. With just five documents, however -- as in this case -- it makes more sense that unexpected words might appear on the 'top features' lists. Overarching training data trends are not necessarily reflected in individual documents.

# ## 4.
# ### General Discussion.
# 
# #### Generally, the model performed better than I expected. However, I'm curious about the gap between the training and testing data -- why is this occurring? To investigate, let's look at the trends in training error and validation error as they relate to the size of the training data.

# In[12]:


def error_scores(X, y, C=10, pen ='l2', slvr ='liblinear', tol=0.1):
    LR = LogisticRegression(C=C, penalty=pen, solver=slvr, tol=tol)
    LR.fit(X,y)
    
    size, scores_trn, scores_val = learning_curve(estimator=LR,
                                                  X=X,
                                                  y=y,
                                                  cv=10,
                                                  scoring="accuracy")
    errors_trn = 1-np.mean(scores_trn, axis=1)
    errors_val = 1-np.mean(scores_val, axis=1)
    output_df  = pd.DataFrame({"Training Size"   : size,
                               "Training Error"  : errors_trn,
                               "Validation Error": errors_val})
    return output_df

all_scores = error_scores(BOW_train_norm,y_train)
scores_val = all_scores['Validation Error']
scores_trn = all_scores['Training Error']
size       = all_scores['Training Size']

sns.set(font_scale = 1)
sns.set_style("whitegrid", {'axes.grid' : False})

plt.plot(size, scores_val, label='Validation Error')
plt.plot(size, scores_trn, label='Training Error')

plt.title ("Validation and Training Error")
plt.xlabel('Training Data Size')
plt.ylabel('Error Rate')
plt.legend()
plt.show()


# #### The plot above shows that validation error -- which represents the model's potential to perform on unseen data -- continually decreases as the size of the training data increases. Meanwhile, the training error is consistently very low. This combination of high variance and low bias indicates overfitting. Given the downward trend of the validation error with increasing training size, the ability of the model to generalize to unseen data could likely be improved with additional data. This overfitting is why there is a difference between the training and testing scores. Moreover, the training and testing data we are given are equal in size. Train/test splits typically result in much more training data than testing data -- i.e., something like an 80/20 split, rather than 50/50. So it seems like we could afford to take some of the current testing data and include it in the training process, because we probably don't need as much data for testing, and the model could benefit from more training examples.
# 
# #### For me, the biggest takeaway from this assignment is that we need multiple ways to measure the model's success, because a single score doesn't capture all of the relevant information.
