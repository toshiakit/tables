MATLAB tables
=============

Playing around with new data type in R2013b - Tables

Tables are rrays in tabular form whose named columns can have different types. It is similar to dataset arrays in Statistics Toolbox, but tables are built into base MATLAB. R2013b also introduced categorical array that handles discrete, nonnumeric data, such as factors. 

Example: Fisher Iris Dataset Classification
-------------------------------------------

Fisher Iris dataset is commonly used dataset in machne learning tutorials, one can build a classifier to predict the type of Iris based on the sepal and petal measurements. 

<pre>
% Load dataset and convert it into a table
clear; clc; close all;
load fisheriris
vars = {'SL','SW','PL','PW','class'};
T = table(meas(:,1),meas(:,2),meas(:,3),meas(:,4),species,...
    'VariableNames',vars);

% Convert the cell array of strings into categorical array
T.class = categorical(T.class);

% Preview the first 5 rows of data
fprintf('Preview the first 5 rows of Fisher Iris dataset\n\n')
disp(T(1:5,:))
</pre>

Preview the first 5 rows of Fisher Iris dataset

<pre>
    SL     SW     PL     PW     class 
    ___    ___    ___    ___    ______

    5.1    3.5    1.4    0.2    setosa
    4.9      3    1.4    0.2    setosa
    4.7    3.2    1.3    0.2    setosa
    4.6    3.1    1.5    0.2    setosa
      5    3.6    1.4    0.2    setosa
</pre>

Check the size of the dataset

<pre>
fprintf('Number of features in the dataset: %d\n', width(T)-1);
fprintf('Number of samples in the dataset: %d\n\n', height(T));
</pre>

Number of features in the dataset: 4<br>
Number of samples in the dataset: 150

Check the number of each class

<pre>
fprintf('Number of each class:\n')
summary(T.class)
</pre>

Number of each class:
<pre>
     setosa          50 
     versicolor      50 
     virginica       50 
</pre>

Check for missing data

<pre>
fprintf('Number of missing data in the dataset: %d\n\n',...
    sum(sum(ismissing(T))));
</pre>

Number of missing data in the dataset: 0



### Classify a new sample with k nearest neighbor (KNN) method ###


We will fit a model using the default parameters and see what class it predicts for a new sample. Prediction Score indicates how confident the prediction is.

<pre>
% Add a new sample
new = [3,5,4,2];

% Fit a KNN classifier
cl = ClassificationKNN.fit(T{:,1:4},cellstr(T.class),...
    'PredictorNames',T.Properties.VariableNames(1:4),...
    'ResponseName',T.Properties.VariableNames{5}...
    );
disp(cl)
</pre>

Inspect the model properties

<pre>
  ClassificationKNN
    PredictorNames: {'SL'  'SW'  'PL'  'PW'}
      ResponseName: 'class'
        ClassNames: {'setosa'  'versicolor'  'virginica'}
    ScoreTransform: 'none'
     NObservations: 150
          Distance: 'euclidean'
      NumNeighbors: 1
</pre>

Make a prediction for the new sample

<pre>
[pred, score] = predict(cl,new);
fprintf('Predicted class of the new sample: "%s"\n', char(pred));
fprintf('Prediction Score: %.2f%%\n\n', score(strcmp(char(pred),...
    cellstr(unique(T.class))))*100);
</pre>


Predicted class of the new sample: "virginica"<br>
Prediction Score: 100.00%

### Quality of KNN Classifier ###

The quality of KNN classifier depends on the choice of parameter k, which specifies the number of nearest neighbors to use in computation. Smaller k is sensitive to noise in the data.

The default is k = 1, which means it assigns the class of the single nearest neighbor. Since we used the whole dataset to fit the classifier, it is going to predict the class of the existing data perfectly, but this is not very useful in practice.

<pre>
rloss = resubLoss(cl);
fprintf('Resubstitution loss at k=1: %.2f%%\n\n', rloss*100);
</pre>

Resubstitution loss at k=1: 0.00%

### Simulate the realistic performnace with cross validation ###

Cross validation enables evaluation of the classifer by withholding a portion of the data during the training, and test the classifier against the withheld data. The average error rate will give us more realisitc idea about the performnace of the classifier.

<pre>
cv = crossval(cl);
kloss = kfoldLoss(cv);
fprintf('Avg cross validation loss at k=1: %.2f%%\n\n', kloss*100);
</pre>

Avg cross validation loss at k=1: 4.00%

### Confusion Matrix ###

Confusion Matrix allows comparison of prediction against the ground truth and serves as the basis to compute the rates of true positives, false positiives, true negatives, and false negatives. Ground truth is mapped to the rows and prediction is mapped to the columns. The diagonal represents the true positivies.

This analysis shows that all misclassifications are between versicolor and virginica.

<pre>
labels = cellstr(unique(T.class));
fprintf('Confusion Matrix for k = %d\n', 2);
cl.NumNeighbors = 2;
pred = resubPredict(cl);
C = confusionmat(cellstr(T.class),pred);
C1 = table(C(:,1),C(:,2),C(:,3),'VariableNames',labels,'RowNames',labels);
disp(C1)
</pre>

Confusion Matrix for k = 2

<pre>
                  setosa    versicolor    virginica
                  ______    __________    _________

    setosa        50         0             0       
    versicolor     0        50             0       
    virginica      0         3            47       
</pre>

Changing k to 3

<Pre>
fprintf('Confusion Matrix for k = %d\n', 3);
cl.NumNeighbors = 3;
pred = resubPredict(cl);
C = confusionmat(cellstr(T.class),pred);
C2 = table(C(:,1),C(:,2),C(:,3),'VariableNames',labels,'RowNames',labels);
disp(C2)
</pre>

Confusion Matrix for k = 3
<pre>
                  setosa    versicolor    virginica
                  ______    __________    _________

    setosa        50         0             0       
    versicolor     0        47             3       
    virginica      0         3            47       
</pre>






