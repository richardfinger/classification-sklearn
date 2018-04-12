# classification-sklearn
Classification into two classes using random forest algorithm.

This script takes the file Training_Full.csv which contains classified
training set as an input. The format of the file is csv where one row is one
observation. The first column of the file is the classification - possible
values are { -1 , 0 , 1 }. The csv file contains a header and all of the data
must be numeric. No values can be missing. During the preprocessing the
problem is reduced to two class problem by merging the -1 and 0 class
together. The script assumes that the classes are imbalanced.

This script shows basic statistics of the data and then performs
 1. Scaling
 2. Feature selection
 3. Grid search on the parameters of random forest
 4. Fit on training data
 5. Display of classification report - compare best random forest to GaussianNB
 6. Print ROC curves into png file

In addition it has the capability to chose a favorable tradeoff between
precision and recall using cutoff threshold

MIT License - Copyrigh (c) 2018 Richard Finger
