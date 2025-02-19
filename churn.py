
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn import metrics, model_selection
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier


%matplotlib inline
customers = pd.read_csv('/content/churn.csv')

# remove target and ID columns because the arbitrary ID data isn't relevant to the analysis
features = customers[customers.columns.difference(
    ['area_code', 'phone_number', 'class'])]

# target
targets = customers[['class']]
# describe the characteristics of variables
features.describe()
# analysis of the dispersion of the features
features.hist(figsize=(12, 10))
plt.show()
plt.hist(
    features.number_customer_service_calls,
    bins=10,
    range=(0, 10),
    color='green')
plt.xlabel('Number Customer Service Calls')
plt.ylabel('Number of Participants')
plt.vlines(
    x=features.number_customer_service_calls.mean(),
    ymin=0,
    ymax=1850,
    linestyles='dashed',
    label="Mean: {:0.0f}".format(
        features.number_customer_service_calls.mean()))
plt.legend(loc=5, fontsize='large')
sns.set_style('white', {'axes.grid': False})
plt.title('Number Customer Service Calls', fontsize='xx-large')
sns.despine()
correlation_matrix = features.corr()
plt.figure(figsize=(15, 10))
ax = sns.heatmap(
    correlation_matrix,
    vmax=1,
    square=True,
    annot=True,
    fmt='.2f',
    cmap='GnBu',
    cbar_kws={"shrink": .5},
    robust=True)
plt.title('Correlation Matrix of features', fontsize=8)
plt.show()
ax = sns.pairplot(features)
plt.title('Pairwise relationships between the features')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.3, random_state=150)

# Random Forest Model
rf = RandomForestClassifier()

# fit the model on your training data
rf.fit(X_train, y_train.values.ravel())

# plot the feature importance
feat_scores = pd.DataFrame(
    {
        'Fraction of Samples Affected': rf.feature_importances_
    },
    index=features.columns)
feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected')
feat_scores.plot(kind='barh', figsize=(8, 4), colormap='rainbow')
sns.despine()
def one_fold_classifier(train_features, train_targets, test_features,
                        test_targets, classifier):

    kfold_result = pd.Series(
        index=['tpr', 'auc', 'acc', 'prec', 'rec'], dtype=object)

    prediction = classifier.fit(train_features, train_targets.values.ravel())
    prediction_prob = prediction.predict_proba(test_features)

    fpr, tpr, t = roc_curve(test_targets, prediction_prob[:, 1])
    y_pred = classifier.predict(test_features)
    mean_fpr = np.linspace(0, 1, 100)

    kfold_result['acc'] = classifier.score(test_features, test_targets)
    kfold_result['prec'] = precision_score(test_targets, y_pred)
    kfold_result['rec'] = recall_score(test_targets, y_pred)
    kfold_result['auc'] = auc(fpr, tpr)
    kfold_result['tpr'] = interp1d(fpr, tpr)(mean_fpr) # Changed interp to interp1d

    return kfold_result

def run_k_fold_classifier(features, target, classifierName, classifier):

    # computes the result obtained for the k folds individualy
    kfold_results = pd.DataFrame()

    # test training and analysis of results for each fold
    for train_index, test_index in cv.split(features, targets.values.ravel()):
        train_features = features.iloc[train_index]
        train_targets = targets.iloc[train_index]
        test_features = features.iloc[test_index]
        test_targets = targets.iloc[test_index]

        currentFold_result = one_fold_classifier(train_features, train_targets,
                                                 test_features, test_targets,
                                                 classifier)
        kfold_results = pd.concat([kfold_results, currentFold_result], ignore_index=True, axis=1) # Changed append to concat due to FutureWarning


    # computes the mean of the results obtained for the individual folds
    kfold_mean_aux = kfold_results.apply(np.mean, axis=0) # Changed axis for apply due to FutureWarning

    kfold_mean_aux['auc'] = auc(mean_fpr, kfold_mean_aux['tpr'])
    kfold_mean_aux['fpr'] = mean_fpr
    kfold_mean[classifierName] = kfold_mean_aux
    folds = 5

# provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds
cv = StratifiedKFold(n_splits=folds, shuffle=False)

# classifiers to be built
classifiersNames = [
    'LogisticRegression', 'RandomForestClassifier', 'AdaBoostClassifier',
    'GradientBoostingClassifier'
]
classifiers = [
    LogisticRegression(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
]

# mean value of all folds
kfold_mean = pd.DataFrame(
    index=['fpr', 'tpr', 'auc', 'acc', 'prec', 'rec'],
    columns=[classifiersNames])

# auc parameter
mean_fpr = np.linspace(0, 1, 100)

# training and test for each of the chosen classifiers
for classifiersName, classifier in zip(classifiersNames, classifiers):
    run_k_fold_classifier(features, targets, classifiersName, classifier)
# plot functions
fig1 = plt.figure(figsize=[10, 5])
for index, classifier in enumerate(classifiersNames):
    print("Model \n", str(classifiersNames[index]))

    print_out_scores = "Scores \n\
        Mean Accuracy: {:0.2f} \n\
        Mean Precision: {:0.2f} \n\
        Mean Recall: {:0.2f} \n\
        AUC: {:0.2f}".format(
        kfold_mean[str(classifiersNames[index])]['acc'],
        kfold_mean[str(classifiersNames[index])]['prec'], kfold_mean[str(
            classifiersNames[index])]['rec'], kfold_mean[str(
                classifiersNames[index])]['auc'])
    print(print_out_scores)
    print("*******\n")

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    color = ['r', 'b', 'g', 'y']
    plt.plot(
        kfold_mean[str(classifiersNames[index])]['fpr'],
        kfold_mean[str(classifiersNames[index])]['tpr'],
        color=color[index],
        label=r'%s  - Mean ROC (AUC = %0.2f )' % (str(
            classifiersNames[index]), kfold_mean[str(
                classifiersNames[index])]['auc']),
        lw=2,
        alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, shuffle=True, test_size=0.2, random_state=15)

# fit Model

model = GradientBoostingClassifier().fit(X_train, y_train.values.ravel())

# R^2 - the best R² possible score is 1.0
print("R²: ", model.score(X_test, y_test))
predictions = model.predict(X_test)
mse = mean_squared_error(y_test.values.ravel(), predictions)
rmse = np.sqrt(mse)
print("rmse: ", rmse)
model = GradientBoostingClassifier()
gb_grid_params = {
    'learning_rate': [0.1, 0.2],
    'max_depth': [4, 5],
    'min_samples_leaf': [5, 7]
}

gs = GridSearchCV(model, gb_grid_params, cv=10, return_train_score=True)

gs.fit(features, targets.values.ravel())
gs.cv_results_
gs.best_estimator_
print("Mean cross-validated score of the best_estimator: ", gs.best_score_ )
