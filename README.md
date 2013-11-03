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











