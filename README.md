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


Example: Boston Housing Prices Regression
------------------------------------------

For regression example, linear regression and a bagged trees methond (Random Forest) are applied to Boston Housing dataset.

The dataset contains 13 demographic features:
 1. Per capita crime rate per town
 2. Proportion of residential land zoned for lots over 25,000 sq. ft.
 3. Proportion of non-retail business acres per town
 4. 1 if tract bounds Charles river, 0 otherwise
 5. Nitric oxides concentration (parts per 10 million)
 6. Average number of rooms per dwelling
 7. Proportion of owner-occupied units built prior to 1940
 8. Weighted distances to five Boston employment centres
 9. Index of accessibility to radial highways
 10. Full-value property-tax rate per $10,000
 11. Pupil-teacher ratio by town
 12. 1000(Bk - 0.63)^2
 13. Percent lower status of the population
 14. [The target] the median value of owner-occupied homes in $1000's.


<pre>
% Load dataset and convert it into a table
clear; clc; close all;
[x,t] = house_dataset;
x = x';
t = t';

vars = {'CRIM','RES','IND','CHAR','NOX','RM','AGE','DIST','RAD','TAX',...
    'PTR','B','LSTAT','Target'};
T = table(x(:,1),x(:,2),x(:,3),x(:,4),x(:,5),x(:,6),x(:,7),x(:,8),...
    x(:,9),x(:,10),x(:,11),x(:,12),x(:,13),t,'VariableNames',vars);

% preview the first 5 rows of data
fprintf('Preview the first 5 rows of Housing Value dataset\n\n')
disp(T(1:5,:))

% Check the size of the dataset
fprintf('Number of features in the dataset: %d\n', width(T)-1);
fprintf('Number of samples in the dataset: %d\n\n', height(T));

% Check for missing data
fprintf('Number of missing data in the dataset: %d\n\n',...
    sum(sum(ismissing(T))));

% Store the current random stream for reproducibility
stream = RandStream.getGlobalStream;
savedState = stream.State;
</pre>

Preview the first 5 rows of Housing Value dataset

<pre>
     CRIM      RES    IND     CHAR     NOX      RM      AGE      DIST     RAD
    _______    ___    ____    ____    _____    _____    ____    ______    ___

    0.00632    18     2.31    0       0.538    6.575    65.2      4.09    1  
    0.02731     0     7.07    0       0.469    6.421    78.9    4.9671    2  
    0.02729     0     7.07    0       0.469    7.185    61.1    4.9671    2  
    0.03237     0     2.18    0       0.458    6.998    45.8    6.0622    3  
    0.06905     0     2.18    0       0.458    7.147    54.2    6.0622    3  


    TAX    PTR       B       LSTAT    Target
    ___    ____    ______    _____    ______

    296    15.3     396.9    4.98       24  
    242    17.8     396.9    9.14     21.6  
    242    17.8    392.83    4.03     34.7  
    222    18.7    394.63    2.94     33.4  
    222    18.7     396.9    5.33     36.2  
</pre>

Number of features in the dataset: 13 <br>
Number of samples in the dataset: 506 <br>
Number of missing data in the dataset: 0<br>

### Histogram the pricing data ###
Check the values in the data

<pre>
figure(1)
hist(T.Target)
xlabel('price($1000s)')
ylabel('count')
</pre>

### Split the dataset into training and test subsets ###

Create an independent test set for checking the regression performance.

<pre>
% Make sure we can reproduce the randomization result
stream.State = savedState;
% Split dataset into 60:40 ratio
part = cvpartition(height(T),'holdout',0.4);
istrain = training(part); % data for fitting
istest = test(part); % data for quality assessment

fprintf('Size of training set: %d\n',sum(istrain))
fprintf('Size of training set: %d\n\n',sum(istest))
</pre>

Size of training set: 304<br>
Size of training set: 202<br>

### Run a simple linear regression ###

There are number of possible algorithms to choose from. We can start with the simplest - linear regression and plot the actual vs. predicted prices to see how it performs.

If the dots line up on a diagonal, then it is doing a good job. But this plot shows that dots are not very tightly clustered along the diagonal, you can also see significant diviation in the higher price range. RMSE indicates that the prediction may be off by as much as $5000 on average.

Note: this is a very naive use of linear regression, since we didn't even normalize the features among many other things.

<pre>
lm = fitlm(T{istrain,1:end-1},T.Target(istrain));
ypred = predict(lm,T{istest,1:end-1});
ytrue = T.Target(istest);
figure(2)
plot(ytrue,ypred,'.')
hold on
plot([0,50],[0,50], '-k')
hold off
xlabel('True price ($1000s)')
ylabel('Predicted price ($1000s)')
xlim([0 50]); ylim([0 50]);
fprintf('RMSE: %f\n\n',sqrt(mean((ytrue-ypred).^2)))
</pre>

RMSE: 5.033373

### Run a bagged trees regression (Random Forest) ###

Try bagged decision trees method to see if it does better. The parameter 'NVarToSample' determines the number of randomly selected features for each decision split, for regression, the default is 1/3 of number of features. This setting invokes Breiman's 'random forest' algorithm.

This more complex model captures the complexity of the dataset better - the dots are now more tightly clustered - but it still have problems in the high price range.

<pre>
btrees = TreeBagger(50,T{istrain,1:end-1},T.Target(istrain),...
    'Method','regression','oobvarimp','on');
ypred = predict(btrees,T{istest,1:end-1});
figure(3)
plot(ytrue,ypred,'.')
hold on
plot([0,50],[0,50], '-k')
hold off
xlabel('True price ($1000s)')
ylabel('Predicted price ($1000s)')
xlim([0 50]); ylim([0 50]);
fprintf('RMSE: %f\n\n',sqrt(mean((ytrue-ypred).^2)))

figure(4)
plot(oobError(btrees))
xlabel('number of grown trees')
ylabel('out-of-bag mean squared error')
</pre>

RMSE: 3.791375

### Variable Importance ###

One very nice feature of bagged trees method, like Random Forest, is that you can get variable importance metrics if you enable 'oobvarimp' option.

It says the most important variable is the number of rooms, then the Percent lower status of the population (whatever that means), etc.

<pre>
varimp = btrees.OOBPermutedVarDeltaError';
vars = T.Properties.VariableNames;
vars(14) =[];
[~,idx]= sort(varimp,'descend');
V = table(varimp(idx),'RowNames',vars(idx),'VariableNames',{'Importance'});
disp(V)
</pre>

Sorted list of variables by Variable Importance Metric

<pre>
             Importance
             __________

    RM         1.5334  
    LSTAT      1.1989  
    CRIM      0.81222  
    NOX       0.67529  
    DIST      0.59925  
    PTR       0.52879  
    IND       0.48142  
    TAX       0.42536  
    AGE       0.39785  
    RES       0.29931  
    B         0.25288  
    RAD       0.17782  
    CHAR     -0.13058  
</pre>

