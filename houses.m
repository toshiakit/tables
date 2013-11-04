%% Boston Housing Values
% Experiment with new Table and Category data types in MATLAB R2013b
% release using Boston Housing dataset.
%
%% Dataset
% The dataset contains 13 demographic features:
%
% # Per capita crime rate per town
% # Proportion of residential land zoned for lots over 25,000 sq. ft.
% # proportion of non-retail business acres per town
% # 1 if tract bounds Charles river, 0 otherwise
% # Nitric oxides concentration (parts per 10 million)
% # Average number of rooms per dwelling
% # Proportion of owner-occupied units built prior to 1940
% # Weighted distances to five Boston employment centres
% # Index of accessibility to radial highways
% # Full-value property-tax rate per $10,000
% # Pupil-teacher ratio by town
% # 1000(Bk - 0.63)^2
% # Percent lower status of the population
%
% The target contains the median value of owner-occupied  homes in $1000's. 

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

clearvars t x vars

%% Histogram the pricing data
% check the values in the data

figure(1)
hist(T.Target)
xlabel('price($1000s)')
ylabel('count')

%% Split the dataset into training and test subsets
% Create an independent test set for checking the regression performance.

% Make sure we can reproduce the randomization result
stream.State = savedState;
% Split dataset into 60:40 ratio
part = cvpartition(height(T),'holdout',0.4);
istrain = training(part); % data for fitting
istest = test(part); % data for quality assessment

fprintf('Size of training set: %d\n',sum(istrain))
fprintf('Size of training set: %d\n\n',sum(istest))

clearvars part

%% Run a simple linear regression
% There are number of possible algorithms to choose from. We can start
% with the simplest - linear regression and plot the actual vs. predicted 
% prices to see how it performs. 
%
% If the dots line up on a diagonal, then it is doing a good job. But this
% plot shows that dots are not very tightly clustered along the diagonal, 
% you can also see significant diviation in the higher price range. RMSE
% indicates that the prediction may be off by as much as $5000 on average.
% 
% Note: this is a very naive use of linear regression, since we didn't
% even normalize the features among many other things.

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

%% Run a bagged trees regression (Random Forest)
% Try bagged decision trees method to see if it does better. The parameter
% 'NVarToSample' determines the number of randomly selected features for
% each decision split, for regression, the default is 1/3 of number of
% features. This setting invokes Breiman's 'random forest' algorithm.
% 
% This more complex model captures the complexity of the dataset
% better - the dots are now more tightly clustered - but it still have
% problems in the high price range. 

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

