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
# features_list = ['poi',
#                 'salary',
#                 'to_messages',
#                 'deferral_payments',
#                 'total_payments',
#                 'exercised_stock_options',
#                 'bonus',
#                 'restricted_stock_deferred',
#                 'restricted_stock',
#                 'shared_receipt_with_poi',
#                 'total_stock_value',
#                 'expenses',
#                 'from_messages',
#                 'other',
#                 'from_this_person_to_poi',
#                 'deferred_income',
#                 'from_poi_to_this_person',
#                 'long_term_incentive',
#                 'total_compensation']



# ### Load the dictionary containing the dataset
# with open("final_project_dataset.pkl", "r") as data_file:
#     data_dict = pickle.load(data_file)

# ### Task 2: Remove outliers
# data_dict.pop('TOTAL')
# data_dict.pop('THE TRAVEL AGENCY IN THE PARK')


# ### Task 3: Create new feature(s)
# for k, v in data_dict.iteritems():
#     if v['total_payments'] != 'NaN' and v['total_stock_value'] != 'NaN':
#         v.update({'total_compensation': float(v['total_payments']) + float(v['total_stock_value'])})
#         str(v['total_compensation'])
#     else:
#         v.update({'total_compensation': '0'})

# features_list.append('total_compensation')



# ### Store to my_dataset for easy export below.
# my_dataset = data_dict

# ### Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, features_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)


#####################################################################
#####################################################################
# Final features list
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'from_poi_to_this_person',
                 'from_poi_rate', 'bonus']

# 0  exercised_stock_options  15.967730
# 2        total_stock_value  15.450997
# 3  from_poi_to_this_person  15.290707
# 4            from_poi_rate  13.270682
# 1                    bonus  13.159868

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0) # Contains column total data
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0) # not an individual
data_dict.pop('LOCKHART EUGENE E', 0) # record contains no information
data_dict.pop('HUMPHREY GENE E', 0) # 'to_poi_rate' outlier
data_dict.pop('LAVORATO JOHN J', 0) # 'from_poi_to_this_person' / 'total_payments' outlier
data_dict.pop('FREVERT MARK A', 0) # 'total_payments' outlier

### Task 3: Create new feature(s)
# for k, v in data_dict.iteritems():
#     if v['total_payments'] != 'NaN' and v['total_stock_value'] != 'NaN':
#         v.update({'total_compensation': float(v['total_payments']) + float(v['total_stock_value'])})
#         str(v['total_compensation'])
#     else:
#         v.update({'total_compensation': '0'})

my_dataset = {}
for key in data_dict:
    my_dataset[key] = data_dict[key]
    try: 
        my_dataset[key]['total_compensation'] = float(data_dict[key]['total_payments'] / \
                             data_dict[key]['total_stock_value'])
    except:
        my_dataset[key]['total_compensation'] = "NaN"

    try:
        my_dataset[key]['from_poi_rate'] = float(data_dict[key]['from_poi_to_this_person'] / \
                             data_dict[key]['to_messages'])
    except:
        my_dataset[key]['from_poi_rate'] = "NaN"

    try:
        my_dataset[key]['to_poi_rate'] = float(data_dict[key]['from_this_person_to_poi'] / \
                           data_dict[key]['from_messages'])
    except:
        my_dataset[key]['to_poi_rate'] = "NaN"


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, remove_NaN=True, remove_all_zeroes=True, 
                     remove_any_zeroes=True, sort_keys=True)

labels, features = targetFeatureSplit(data)
#####################################################################
#####################################################################



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# # Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)




from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

## Testing naive Bayes
# clf = GaussianNB()
# clf = DecisionTreeClassifier()
# clf = RandomForestClassifier()

# 1st test with DT
# clf = DecisionTreeClassifier(criterion='gini', max_depth=5,
#             max_features=4, min_samples_leaf=1,
#             min_samples_split=5, presort=False, splitter='best')


# 2nd test with DT
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(max_features=2, min_samples_split=2,
#                              criterion='entropy', max_depth=None)


# 3rd test with DT
clf = DecisionTreeClassifier(max_features=3, min_samples_split=3,
                             criterion='gini', max_depth=None)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

