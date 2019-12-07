#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Support packages

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'exercised_stock_options',
                 'total_stock_value',
                 'from_poi_to_this_person',
                 'bonus',
                 'salary']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')
data_dict.pop('WHALEY DAVID A')
data_dict.pop('GRAMM WENDY L')
data_dict.pop('WROBEL BRUCE')
data_dict.pop('LAY KENNETH L')


### Task 3: Create new feature(s)
for k, v in data_dict.iteritems():
    if v['total_payments'] != 'NaN' and v['total_stock_value'] != 'NaN':
        v.update({'total_compensation': float(v['total_payments']) + float(v['total_stock_value'])})
        str(v['total_compensation'])
    else:
        v.update({'total_compensation': '0'})

features_list.append('total_compensation')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

## GaussianNB
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(features, labels)


## DecisionTree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)





# from sklearn.model_selection import KFold
# kf = KFold(n_splits = 4, shuffle = True, )
#
# for train_indices, test_indices in kf.split(labels):
#     # make training and testing dataset
#     kfeatures_train = [features[ii] for ii in train_indices]
#     kfeatures_test = [features[ii] for ii in test_indices]
#     klabels_train = [labels[ii] for ii in train_indices]
#     klabels_test = [labels[ii] for ii in test_indices]


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

