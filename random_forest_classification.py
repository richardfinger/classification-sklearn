#!/usr/bin/env python3
#
# Classification into two classes using random forest algorithm.
#
# This script takes the file Training_Full.csv which contains classified
# training set as an input. The format of the file is csv where one row is one
# observation. The first column of the file is the classification - possible
# values are { -1 , 0 , 1 }. The csv file contains a header and all of the data
# must be numeric. No values can be missing. During the preprocessing the
# problem is reduced to two class problem by merging the -1 and 0 class
# together. The script assumes that the classes are imbalanced.
#
# This script shows basic statistics of the data and then performs
#  1. Scaling
#  2. Feature selection
#  3. Grid search on the parameters of random forest
#  4. Fit on training data
#  5. Display of classification report - compare best random forest to GaussianNB
#  6. Print ROC curves into png file
#
# In addition it has the capability to chose a favorable tradeoff between
# precision and recall using cutoff threshold
#
# MIT License - Copyrigh (c) 2018 Richard Finger

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV

# Switch backend to print png into a file
plt.switch_backend('agg')


def print_roc_png(pipeline, y_test, y_test_pred_prob):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    classes = pipeline.named_steps['model'].classes_
    n_classes = classes.shape[0]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test.values, y_test_pred_prob[:, 1])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    filename = type(pipeline.named_steps['model']).__name__ + "_ROC.png"
    plt.savefig(filename)
    plt.clf()


def threshold_proba(proba, classes, threshold):
    # Produce list of classes based on probabilities
    # Assumes that there are two classes in the data 0 and 1
    # If probability > threshold then data point is in class 1
    retval = []
    if classes[0] == 1:
        index = 0
    if classes[1] == 1:
        index = 1
    proba = proba[:, index]
    for i in range(proba.shape[0]):
        if proba[i] > threshold:
            retval.append(1)
        else:
            retval.append(0)
    return retval


def features_from_file(X):
    # Select features based on a file containing indices of columns
    raw_indices = np.genfromtxt('Features.csv', delimiter=',')
    indices = tuple(raw_indices.astype(int))
    return X[:, indices]


def scoring_averaged(estimator, x, y):
    # Scoring metric
    y_pred = estimator.predict(x)
    return f1_score(y, y_pred, average='macro')


def print_df_statistics(df):
    print("Number of columns: ", df.shape[1])
    print("Number of rows: ", df.shape[0])
    print("Data types: ", df.dtypes.unique())
    print("Data type counts: ", list(df.dtypes.value_counts()))
    print("Data is missing values: ", df.isnull().values.any())
    print("There are the following classes: ", df.iloc[:, 0].unique())
    print("With value counts: ", list(df.iloc[:, 0].value_counts()))
    print("With percentages : ", [x/df.shape[0]
                                  for x in list(df.iloc[:, 0].value_counts())])


def main():
    df = pd.read_csv('Training_Full.csv')

    # Merge classes -1 and 0 into one class labelled by 0
    for i in range(df.shape[0]):
        if df.iloc[i, 0] == -1:
            df.iloc[i, 0] = 0

    print("** Data statistics after merging -1 and 0 class into 0 class ** ")
    print_df_statistics(df)

    data = df.iloc[:, 1:]
    target = df.iloc[:, 0]

    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.5)

    # Feature selection
    scaler = StandardScaler()

    # Random forest is set with some initial parameters
    # I assume that the feature importance does not change with different parameters of the forest
    model = RandomForestClassifier(
        class_weight='balanced', max_depth=20, max_features=0.8, min_samples_split=10, n_jobs=-1)
    pipeline_for_feature_selector = Pipeline(
        [('scaler', scaler), ('model', model)])

    pipeline_for_feature_selector.fit(x_train, y_train)
    importances = pipeline_for_feature_selector.named_steps['model'].feature_importances_
    indices = np.argsort(importances)[::-1]

    # Number of features selected depends on the threshold
    feature_selector = SelectFromModel(
        pipeline_for_feature_selector.named_steps['model'], threshold=0.0012, prefit=True)
    feature_indices = feature_selector.get_support(True)
    np.savetxt('Features.csv', feature_indices, fmt='%g', delimiter=',')

    # Grid search
    scaler = StandardScaler()

    feature_selector = FunctionTransformer(features_from_file)

    model = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
    pipeline = Pipeline(
        [('scaler', scaler), ('feature_selector', feature_selector), ('model', model)])

    param_grid = [{'model__n_estimators': [20, 30],
                   'model__min_samples_leaf': [0.00001, 0.00005, 0.0001, 0.0005],
                   'model__max_features': [0.15, 0.3, 0.5]}]

    search = GridSearchCV(pipeline, cv=5, param_grid=param_grid,
                          n_jobs=1, scoring=scoring_averaged, refit=True)
    search.fit(x_train, y_train)

    # Training and prediction
    scaler = StandardScaler()

    feature_selector = FunctionTransformer(features_from_file)

    model = GaussianNB()
    pipeline_naive = Pipeline(
        [('scaler', scaler), ('feature_selector', feature_selector), ('model', model)])
    pipeline_naive.fit(x_train, y_train)

    # This pipeline is already fitted from the grid search
    pipeline_forest = search.best_estimator_

    threshold = 0.5

    pipelines = [pipeline_naive, pipeline_forest]

    for pipeline in pipelines:

        print("Pipeline :", pipeline.get_params)
        print("Model :", pipeline.named_steps['model'].get_params())
        if 'feature_selector' in pipeline.named_steps:
            print("*****  Feature selector****")
            print("Number of features:",
                  pipeline.named_steps['feature_selector'].transform(data).shape[1])

        print("*** Train data ***")
        y_train_pred = pipeline.predict(x_train)

        classes = pipeline.named_steps['model'].classes_
        y_train_pred_prob = pipeline.predict_proba(x_train)
        y_train_pred_from_prob = threshold_proba(
            y_train_pred_prob, classes, threshold)

        print(" -- Prediction ")
        print(classification_report(y_train.values, y_train_pred))
        print("F1 Score:", f1_score(y_train.values, y_train_pred, average='macro'))

        print(" -- Prediction based on threshold")
        print(classification_report(y_train.values, y_train_pred_from_prob))
        print("F1 Score:", f1_score(y_train.values,
                                    y_train_pred_from_prob, average='macro'))

        print("*** Test data ***")
        y_test_pred = pipeline.predict(x_test)

        classes = pipeline.named_steps['model'].classes_
        y_test_pred_prob = pipeline.predict_proba(x_test)
        y_test_pred_from_prob = threshold_proba(
            y_test_pred_prob, classes, threshold)

        print(" -- Prediction ")
        print(classification_report(y_test.values, y_test_pred))
        print("F1 Score:", f1_score(y_test.values, y_test_pred, average='macro'))

        print(" -- Prediction based on threshold")
        print(classification_report(y_test.values, y_test_pred_from_prob))
        print("F1 Score:", f1_score(y_test.values,
                                    y_test_pred_from_prob, average='macro'))

        print_roc_png(pipeline, y_test, y_test_pred_prob)


main()
