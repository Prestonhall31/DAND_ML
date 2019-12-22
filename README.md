# Data Analysis - Machine Learning
Preston Hall 
November 2019



The goal of this project is to train a supervised maching learning algorithm using data obtained from the Enron Scandal data set. This data set contains financial information and email data from various employees in the former company. Enron was an energy commodities and services corporation that went bankrupt in 2001 due to fraud. You can learn more about the scandal here. 

### 1. Data Exploration
>Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?


This dataset was provided in a pickle file, which I then extracted using Python's Pickle package. I converted the dataset to a pandas dataframe for easier manipulation and data cleaning. 

Using pandas `.info()` method I could set that there were 146 rows(names) and 21 columns(attributes). The attributes included:  
>salary  
to_messages  
deferral_payments  
total_payments  
exercised_stock_options  
bonus  
restricted_stock  
shared_receipt_with_poi  
restricted_stock_deferred  
total_stock_value  
expenses  
loan_advances  
from_messages  
other  
from_this_person_to_poi  
poi  
director_fees  
deferred_income  
long_term_incentive  
email_address  
from_poi_to_this_person  

The 'poi' attribute is a boolean value that tells us whether this individual was involved in the Enron scandal. Using machine learning algorithms, we can spot patterns across multiple variables that enable more accurate classification. 

Since this was a low enough amount, I chose to print the names and review for anything data that might look out of place. The rows, 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' are not individuals so I decided to remove them from the dataset.

Missing data can make it difficult to have accurate results when running algorithms. I checked which rows had mostly null values and removed all rows with more than 85% null values, which turned out to be: 

```
GRAMM WENDY L has 85.71% null values, poi =  False
LOCKHART EUGENE E has 95.24% null values, poi =  False
THE TRAVEL AGENCY IN THE PARK has 85.71% null values, poi =  False
WHALEY DAVID A has 85.71% null values, poi =  False
WROBEL BRUCE has 85.71% null values, poi =  False
```


I used a scatterplot chart to show the correlation between total_payments and total_stock_value. I could see that there was a datapoint that was clearly an outlier. Sorting the chart by 'total_payments', I could see that this was 'LAY KENNETH L' with 103,559,793.00. The next highest was 'FREVERT MARK A' with 17,252,530.00. I decided to remove 'LAY KENNETH L' so it would not alter the output too much. 


## 2. Feature selection

> What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.


Since 'total_payments' and 'total_stock_value' are both ways to compensate an individual, I decided to combine the two features to create a new feature called 'total_compensation'. This value fully emcompasses how an employee was compensated and could provide more information as to whether the person could be considered a 'poi'.

After adding 'total_compensation' to my features_list, I ran the data using SKLearns SelectKBest algorithm to return the best features. SelectKBest retains the first k features of X with the highest scores.  The f_classif argument I decided to keep the 4 best features to use in my analysis. 

I was having a difficult time getting a good Precision and Recall score based on the features that I had so I went back a created a few more to see if I can increase the score. I created `'from_poi_rate'` and `'to_poi_rate'`. After running some tests, the scores increased quite a bit so I decided to add them into my analysis. 

SelectKBest returned the following features and their f-scores. 

| Feature_names  |  F_Scores |
| --- | --- |
| exercised_stock_options | 15.967730 |
|  total_stock_value | 15.450997 |
| from_poi_to_this_person |  15.290707 |
| from_poi_rate | 13.270682 |
|  bonus |  13.159868 |


I can see that the feature I created, `'total_compensation'`, did not make the cut so I will not be using it in my analysis. 


## 3. Algorithm Selection

> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

I reviewed the documentation on using Pipelines and learned that I can run many algorithms with a few lines of code and see what the output would be on the default values. I started off with running 6 different classifiers to see what the model score wouled return. 

I used several classifiers to see what results would be returned using only the default parameters. 


| Classifier | Accuracy | Precision | Recall | 
| --- | --- | --- | --- |
| GaussianNB |  0.83731  | 0.46254 | 0.35500 |
| DecisionTreeClassifier |  0.79292 | 0.33253 | 0.34350
| RandomForestClassifier  | 0.893 | 0.52193 | 0.23800 |


GaussianNB(priors=None)
        Accuracy: 0.83731       Precision: 0.46254      Recall: 0.35500 F1: 0.40170     F2: 0.37231
        Total predictions: 13000        True positives:  710    False positives:  825   False negatives: 1290   True negatives: 10175



>**Gaussian Naive Bayes:** A Gaussian Naive Bayes algorithm is a special type of NB algorithm. It's specifically used when the features have continuous values. It's also assumed that all the features are following a gaussian distribution i.e, normal distribution.  
<br>
**Decision Tree:** A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.  
<br>
**Random Forest:** Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean prediction of the individual trees.  
 

I decided to use DecisionTreeClassifier in my analysis as it returned really great scores using the default parameters and it allows for multiple varaiations in the parameters to try to improve the scores. 

## 4. Tuning

> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).


Tuning is essentially selecting the best parameters for an algorithm to optimize its performance. Sklearn DecisionTree has several optional arguments that can be passed through to increase the effectiveness of the algorithm. From the documentation page, the arguments and their default values are:

- class_weight=None 
- criterion='gini' 
- max_depth=None
- max_features=None 
- max_leaf_nodes=None
- min_impurity_split=1e-07 
- min_samples_leaf=1
- min_samples_split=2 
- min_weight_fraction_leaf=0.0
- presort=False
- random_state=None 
- splitter='best'

It is possible to overfit the data which can lower our accuracy and precision so it is important to not overfit the data. 

After trying many different parameters, the best collection that I could find were these:

The first set of parameters I attempted were:

```python
clf = DecisionTreeClassifier(max_features=3, min_samples_split=3,
                             criterion='gini', max_depth=None)
```

Which returned: 

> Accuracy: 0.80815            
Precision: 0.36045       
Recall: 0.31900

This is already an improvement over the default parameters. I want to see if I can improve the scores even more. 


```python
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features=4, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=5, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')

```

> Accuracy: 0.80454            
Precision: 0.32216      
Recall: 0.24500

The accuracy stayed the same but the Precision and Recall scores dropped significantly. I feel I may be overfitting the data. I removed the additional parameters and decreased max_features and min_samples_split and changed the criterion to 'entropy' to see if that will help my scores. 

```python
clf = DecisionTreeClassifier(max_features=2, min_samples_split=2,
                             criterion='entropy', max_depth=None)

```

> Accuracy: 0.80515            
Precision: 0.36298      
Recall: 0.35300

This returned the best overal scores. I will use these features set of parameters in my final algorithm. 


## 5. Validation
> What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?


Validation is the process of retaining a sample of the data set and using it to test the machine learning algorithm once it has been tuned and trained. The validation process helps prevent overfitting the algorithm and thus decrease the accuracy of our results. 

Working with small data sets such as this, the data sampling process that creates the training and testing sets can have a siginficant import on the performance. To help prevent decrease in performance, I used SkLearn's train_test_split function. The train_test_split function will split the arrays or matrices into random train and test subsets. 

I validated my data by splitting the data set using train_test_split with a test size of 30%.


## 6. Evaluation

> Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

I ran each classifer against the `tester.py` script which returns the Accracy, precision, recall and F scores. The three evaluation metrics I focused on were accuracy, precision, and recall. My average performace for those metrics with Naive Bayes Decision Tree and Random Forest algorithms are the following:

| Classifier | Accuracy | Precision | Recall | 
| --- | --- | --- | --- |
| GaussianNB |  0.83731  | 0.46254 | 0.35500 |
| DecisionTreeClassifier |  0.80515  | 0.36298   | 0.35300 |
| RandomForestClassifier  | 0.89300 | 0.52193 | 0.23800 |

**Accuracy:** Is the number of correct predictions divided by the total number of predictions.

**Precision:** Is the fraction of relevant instances among the retrieved instances.

**Recall:** Is the fraction of the total amount of relevant instances that were actually retrieved. 

Each metric returns a score between 0 and 1 with 1 being the highest. 

As you can see by the results table, GaussianNB returned the best overall scores though I used Decision Tree in my algorithm as it would allow me to adjust the parameters. 

## Conclusion


This was a great introductory project into Machine Learning algorithms as it allows the exploration of the different processes and methods available and to compare the results to see which method may be more efficient. With the right combonation of features, algorithms and their parameters, we are able to predict the fraudsters at Enron using the data provided. 


### Resources

Sklearn package library [https://scikit-learn.org/stable]

Udacity [https://classroom.udacity.com]

Quora [quora.com/How-do-I-properly-use-SelectKBest-GridSearchCV-and-cross-validation-in-the-sklearn-package-together]

Using pipelines [https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf]

Accuracy [https://developers.google.com/machine-learning/crash-course/classification/accuracy]

Precision and Recall [https://en.wikipedia.org/wiki/Precision_and_recall]

