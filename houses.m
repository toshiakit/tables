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

% Preview the first 5 rows of data
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

hist(T.Target)
xlabel('price($1000s)')
ylabel('count')

%% Split the dataset into training and test subsets
% 

% Make sure we can reproduce the randomization result
stream.State = savedState;
% Split dataset into 60:40 ratio
[train,test] = crossvalind('holdout',height(T),0.4);

fprintf('Size of training set: %d\n',sum(train))
fprintf('Size of training set: %d\n\n',sum(test))

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

lm = fitlm(T{train,1:end-1},T.Target(train));
ypred = predict(lm,T{test,1:end-1});
ytrue = T.Target(test);
figure(1)
plot(ytrue,ypred,'.')
hold on
plot([0,50],[0,50], '-k')
hold off
xlabel('True price ($1000s)')
ylabel('Predicted price ($1000s)')
xlim([0 50]); ylim([0 50]);
fprintf('RMSE: %f\n\n',sqrt(mean((ytrue-ypred).^2)))

%% Run a bagged trees regression
% Try bagged decision trees method to see if it does better. As you can
% see, this more complex model captures the complexity of the dataset
% better - the dots are now more tightly clustered - but it still have
% problems in the high price range. 

btrees = TreeBagger(50,T{train,1:end-1},T.Target(train),...
    'Method','regression','oobvarimp','on');
ypred = predict(btrees,T{test,1:end-1});
figure(2)
plot(ytrue,ypred,'.')
hold on
plot([0,50],[0,50], '-k')
hold off
xlabel('True price ($1000s)')
ylabel('Predicted price ($1000s)')
xlim([0 50]); ylim([0 50]);
fprintf('RMSE: %f\n\n',sqrt(mean((ytrue-ypred).^2)))

%% Variable Importance 
% One very nice feature of bagged trees method is that you can get variable
% importance metrics if you enable 'oobvarimp' option.
%
% So it says the most important variable is the number of rooms, then the
% Percent lower status of the population (whatever that means), etc.

varimp = btrees.OOBPermutedVarDeltaError';
vars = T.Properties.VariableNames;
vars(14) =[];
[~,idx]= sort(varimp,'descend');
V = table(varimp(idx),'RowNames',vars(idx),'VariableNames',{'Importance'});
disp(V)
