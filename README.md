# Data Analysis - Machine Learning
Preston Hall 
November 2019

## 1. Introduction 

> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?**

The goal of this project is to train a supervised maching learning algorithm using data obtained from the Enron Scandal. This data set contains financial information and email data from various employees in the former company. Enron was an energy commodities and services corporation that went bankrupt in 2001 due to fraud. You can learn more about the scandal here. This ML project will attempt to classify whether an Enron employee was a person of interest (POI).


## Data Exploration

I converted the dataset to a pandas dataframe for easier manipulation and cleaning. Using `df.info()` I could set that there were 146 rows(names) and 21 columns(attributes). Since this was a low enough amount, I printed the names and reviewed for anything that might stick out to me. The rows, 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' stuck out particularly as these are not people. I decided to remove them from the dataset. 

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

SelectKBest returned the following features and their f-scores. 

| Feature_names  |  F_Scores |
| --- | --- |
| exercised_stock_options | 15.967730 |
|  total_stock_value | 15.450997 |
| from_poi_to_this_person |  15.290707 |
|  bonus |  13.159868 |

I can see that the feature I created, `'total_compensation'`, did not make the cut so I will not be using it in my analysis. 


## 3. Algorithm Selection

> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?


I tried 2 algorithms to see which ones would have the best effect on this dataset. I used Gaussian Naive Bayes and Decision Tree with only the default arguments to see what the results would be. Neither of these algorithms require feature scaling, so I used the original data values.

```
GaussianNB: 
    Accuracy: 0.82254 
    Precision: 0.38570
    Recall: 0.25900

DecisionTree: 
    Accuracy: 0.76262
    Precision: 0.22154
    Recall: 0.21600

RF: Accuracy: 0.84429, Precision: 0.40405, Recall: 0.18950
```

GaussianNB provided higher accuracy upfront but I decided to work with Decision Tree to see if I can get the score higher, as there are more optional arguments to work with which I will explain in the next section.

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

```
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features=4, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=5, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')

```

```
Accuracy: 0.77800             
Precision: 0.26831       
Recall: 0.25650
```

I'm sure that if I cleaned up the data more or chose different features, I may be able to increase the scores of the data. 


## 5. Validation
> What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?


Validation is the process of retaining a sample of the data set and using it to test the machine learning algorithm once it has been tuned and trained. The validation process helps prevent overfitting the algorithm and thus decrease the accuracy of our results. 



With small data sets such as the Enron set, the data sampling process that creates the training, test and validation sets can have a significant impact on the classifier’s performance – for example, if the distribution of data in the training set does not reflect that of the wider set. To overcome this, I used a cross-validation function, which randomly splits the data into k samples and trains the classifier on each of the k-1 samples, before validating it on the remaining data. The classifier’s performance is thus averaged across each of the samples.

The specific function I used (StratifiedShuffleSplit) has the additional benefit of stratifying each random sample, such that the distribution of classes (i.e. POI and non-POI) in each sample reflects that of the larger data set. This is important, particularly in such a small and unevenly distributed data set, because otherwise there is no guarantee that each sample being used to train the classifier actually contains POI data for it to learn from.


## 6. Evaluation

> Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.
