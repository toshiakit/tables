%% Ensemble Methods
% Experiment with new Table and Category data types in MATLAB R2013b
% release using various datasets. This is based on this example:
%
% <http://www.mathworks.com/help/stats/ensemble-methods.html>
%
%% Fisher Iris Dataset
% Features in the dataset are the measurements of iris flowers. The goal
% is to predict the type of iris based on average measurements. 

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

% Check the size of the dataset
fprintf('Number of features in the dataset: %d\n', width(T)-1);
fprintf('Number of samples in the dataset: %d\n\n', height(T));

% Check the number of each class
fprintf('Number of each class:\n')
summary(T.class)

% Check for missing data
fprintf('Number of missing data in the dataset: %d\n\n',...
    sum(sum(ismissing(T))));

clearvars ans meas species vars

%% Create an ensemble for classification
% The syntax of creating ensemble object is 
% 
%   ens = fitensemble(X,Y,model,numberens,learners)
% 
% * X: matrix of features with observations
% * Y: vector of responses with the same number of observations
% * model: type of ensembe - we will use 'AdaBoostM2'
% * numberens: number of each learners - use 100 for this example
% * learners: learning alogirhtms to use - use default tree template
% 

ens = fitensemble(T{:,1:4},T.class,'AdaBoostM2',100,'Tree',...
    'PredictorNames',T.Properties.VariableNames(1:4),...
    'ResponseName',T.Properties.VariableNames{5},...
    'ClassNames',cellstr(unique(T.class))...
    );
disp(ens)
flower = predict(ens,mean(T{:,1:4}));
fprintf('Predicted class of average measurement: %s\n\n', char(flower));

%% Cars dataset
% The dataset contains measurements of cars in 1970, 1976, and 1982.
% The goal is to predict MPG based on Horsepower and Weight for a car 
% with 150 horsepower weighing 2750 lbs.

% Load dataset and convert it into a table
clear; clc; close all;
load carsmall
T = table(Model,Model_Year,Acceleration,Cylinders,Displacement,...
    Horsepower,MPG,Mfg,Origin,Weight);

% Convert the cell array of strings into categorical array
T.Model = cellstr(T.Model);
T.Model_Year = categorical(T.Model_Year);
T.Mfg = categorical(cellstr(T.Mfg));
T.Origin = categorical(cellstr(T.Origin));

% Preview the first 5 rows of data
fprintf('Preview the first 5 rows of Cars dataset\n\n')
disp(T(1:5,:)) 

% Check the size of the dataset
fprintf('Number of features in the dataset: %d\n', width(T));
fprintf('Number of samples in the dataset: %d\n', height(T));

% Check for missing data
fprintf('Number of missing data in the dataset: %d\n',...
    sum(sum(ismissing(T))));
fprintf('Missing data are in variables:\n')
disp(T.Properties.VariableNames(any(ismissing(T))))
% Find the indices of good data
good = ~any(ismissing(T),2);

clearvars -except T good

%% Create an ensemble for regression
% For this dataset, LSBoost ensemble type is used for regression. 
% Then predict the mileage of a car with 150 horsepower weighing 2750 lbs.
X = [T.Horsepower T.Weight];
ens = fitensemble(X(good,:),T.MPG(good),'LSBoost',100,'Tree',...
    'PredictorNames',T.Properties.VariableNames([6 10]),...
    'ResponseName',T.Properties.VariableNames{7}...   
    );
disp(ens)
mileage = ens.predict([150 2750]);
fprintf('Predicted mileage of the car is: %.2f MPG\n\n', mileage);

%% Evaluating Ensemble Quality
% Ensembles tend to overfit the data, it is essential to evaluate the
% quality of the ensemble. 
%
% In this example we will use a randomly generated dataset and apply three
% different methods. 

% Generate a dataset of 2000 observations with 20 features. 
clear; clc; close all;
rng(1,'twister') % for reproducibility
X = rand(2000,20);
% Generate target labels - if sum of the first 5 columns in each row is
% greater than 2.5, then the label it 1, else label it 0. 
Y = sum(X(:,1:5),2) > 2.5;
% Randomly switch 10% of the labels out of 2000 observations
idx = randsample(2000,200); % pick 10%
Y(idx) = ~Y(idx); % swtich 0 to 1 and 1 to 0

clearvars idx

%% Method 1: Evaluation by Indenpendent Test Set
% One way to evaluate the quality of the ensemble is to split the dataset
% into training and test sets. By plotting the classification error against 
% the number of trees, you can see that 30 is the optimal number of trees 
% to use.    

% Holdout 30% of the dataset for testing and use 70% for training
cvpart = cvpartition(Y,'holdout',0.3);
Xtrain = X(training(cvpart),:);
Ytrain = Y(training(cvpart),:);
Xtest = X(test(cvpart),:);
Ytest = Y(test(cvpart),:);

% Create a bagged ensembe of 200 classification trees
bag = fitensemble(Xtrain,Ytrain,'Bag',200,'Tree',...
    'type','classification');
disp(bag)

% Plot the classification errors against the number of trees
figure(1);
plot(loss(bag,Xtest,Ytest,'mode','cumulative'));
xlabel('Number of trees');
ylabel('Classification error');
legend('Test','NE');

%% Method 2: Cross Validation
% k-fold cross validation splits the dataset into k different subsets, and
% hold out one fold to train the classifier on the test, and this is
% repeated over different combinations of folds. Errors are calculated with
% the heldout fold each time, and the results are averaged. 
% 
% In this case the cross validation classification errors are lower than
% the test case, and the optimal number of trees also shift higher.

% Create a bagged ensembe of 200 classification trees with 5-fold cross
% validation using the entire dataset. 
cv = fitensemble(X,Y,'Bag',200,'Tree',...
    'type','classification','kfold',5);
disp(cv)

% Plot the classification errors against the number of trees
hold on;
plot(kfoldLoss(cv,'mode','cumulative'),'r.');
hold off;
legend('Test','Cross-validation','Location','NE');

%% Method 3: Out of bag Estimate
% This is similar to k-fold cross validation, but it uses boostrapping or
% booosting instead. The observations are randomly sampled with
% replacement, meaning that each subset may contain overlapping
% observations. 
%
% In this case, OOB errors are significantly higher, but the optimal number
% of trees seems to be comparable to that of cross validation method.

% Plot the classification errors against the number of trees
hold on;
plot(oobLoss(bag,'mode','cumulative'),'k--');
hold off;
legend('Test','Cross-validation','Out of bag','Location','NE');

%% Classification with Imbalanced Data with RUSBoost
% In the real world, we don't always get a well balanced dataset that has
% roughly equal number of observations in each class. In some cases, the
% observations may be skewed extremely for one class over the others. 
% RUSBoost is especially effective at classifying such imbalanced data.
% 
% For this example, download the Forest Cover dataset from
%
% <http://archive.ics.uci.edu/ml/datasets/Covertype>
%
% The dataset shows imbalance among the classes, and therefore it is a
% good candidate for RUSBoost.

clear; clc; close all;
load 'covtype.data'
Y = covtype(:,end);
X = covtype(:,1:end-1);
fprintf('Check the balance among casses\n');
tabulate(Y)

clearvars covtype

%% Partion the dataset into training and test sets and fit the model
% Since we have a very large dataset, we can split into 50/50 to speed up
% the training and also evaluate the result with independent test. 

part = cvpartition(Y,'holdout',0.5);
istrain = training(part); % data for fitting
istest = test(part); % data for quality assessment
fprintf('\nCheck the balance of training set after partitioning\n');
tabulate(Y(istrain))

% Uss the classificaiton tree template where each leaf of the trees has at
% least 5 observations. 
t = ClassificationTree.template('minleaf',5);
fprintf('\nStart training the RUSBoost ensemble\n');
tic
% number of trees = 1000, and learning rate = 0.1, with print out to track
% the progress for every 100 trees trained
rusTree = fitensemble(X(istrain,:),Y(istrain),'RUSBoost',1000,t,...
    'LearnRate',0.1,'nprint',100);
toc

%% Classification Error Plot with RUSBoost
% As before, we plot the classification error using the test set against 
% the number of trees to visualize the classification performance. It
% achieves the error rate of 24% around trees = 400 or more.

figure(2);
tic
plot(loss(rusTree,X(istest,:),Y(istest),'mode','cumulative'));
toc
grid on;
xlabel('Number of trees');
ylabel('Test classification error');

%% Confusion Matrix with RUSBoost
% Confusion Matrix shows that Classes 3 to 7 all have > 90% accuracy, and
% the majority of errors come from misclassificaiton of class 2. 

tic
Yfit = predict(rusTree,X(istest,:));
toc
tab = tabulate(Y(istest));
CM = bsxfun(@rdivide,confusionmat(Y(istest),Yfit),tab(:,2))*100;

fprintf('Comfusion Matrix\n');
disp(CM)
fprintf('Diagonal of Confusion Matrix\n');
disp(diag(CM))

%% Reduce the size of RUSBoost ensemble
% Ensembles grows a lot of trees in the training process, and that grows
% the size of the ensemble. You can reduce the size by reducing the number
% of trees it holds, and you can still get pretty good performance. 

% Compact the ensemble
cmpctRus = compact(rusTree);
sz(1) = whos('rusTree');
sz(2) = whos('cmpctRus');
fprintf('Compating before and after\n');
disp([sz(1).bytes sz(2).bytes])

% Remove half of the trees
cmpctRus = removeLearners(cmpctRus,500:1000);
sz(3) = whos('cmpctRus');
fprintf('Removed half of the trees\n');
disp(sz(3).bytes)

% Check the loss rate after reduction
L = loss(cmpctRus,X(istest,:),Y(istest));
fprintf('Overall loss rate after reduction: %.2f%%\n\n', L*100);

%% Classification with Imbalanced Data with Unqual Misclassification Cost
% In practice, the misclassificaiton cost can differ substantially across
% different classes, as in the case of medical diagnosis, where
% misdiagnosis of serious disease (false negative) may be very costly. How
% do we deal with this situation?
% 
% Option 1: if certain classes are under- or overrepresented in the 
% dataset, then adjust the prior probability through applying different 
% sampling rates.
% Option 2: If the classes are represented appropriately represented in the
% dataset, but you want to treat them asymmetrically, then use cost
% parameter. 
% 
% Predict survival probability with Heptitis dataset is available from
% 
% <http://archive.ics.uci.edu/ml/datasets/Hepatitis>
%

clear; clc; close all;
s = urlread(['http://archive.ics.uci.edu/ml/' ...
  'machine-learning-databases/hepatitis/hepatitis.data']);
fid = fopen('hepatitis.txt','w');
fwrite(fid,s);
fclose(fid);
VarNames = {'die_or_live' 'age' 'sex' 'steroid' 'antivirals' 'fatigue' ...
    'malaise' 'anorexia' 'liver_big' 'liver_firm' 'spleen_palpable' ...
    'spiders' 'ascites' 'varices' 'bilirubin' 'alk_phosphate' 'sgot' ...
    'albumin' 'protime' 'histology'};
T = readtable('hepatitis.txt',...
    'Delimiter',',','ReadVariableNames',false,'TreatAsEmpty','?',...
    'Format','%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f');
T.Properties.VariableNames = VarNames;

% Preview the first 5 rows of data
fprintf('Preview the first 5 rows of Hepatitis dataset\n\n')
disp(T(1:5,:)) 

ClassNames = {'Die' 'Live'};
T.die_or_live = ClassNames(T.die_or_live)';

% Convert the cell array of strings into categorical array
T.die_or_live = categorical(T.die_or_live);

% Check the size of the dataset
fprintf('Number of features in the dataset: %d\n', width(T)-1);
fprintf('Number of samples in the dataset: %d\n\n', height(T));

% Check the number of each class
fprintf('Number of each class:\n')
summary(T.die_or_live)

% Check for missing data
fprintf('Number of missing data in the dataset: %d\n',...
    sum(sum(ismissing(T))));
fprintf('Max faction of missind data per variable: %.2f%%\n\n',...
    max(sum(ismissing(T))/height(T))*100)

clearvars ans fid s VarNames

%% Use Surrogate Splits to deal with Missing Data
% Because of the large portion of the data contains some missing data, we
% can't simply throw away the bad observations. Surrgate splits solves this
% problem, and it won't be too slow becuase of the small size of dataset.

rng(0,'twister') % for reproducibility
t = ClassificationTree.template('surrogate','all');

%% Classification without Cost Adjustment
% For binary classification with multi-level predictors, we use
% GentleBoost with 150 trees, and we specify the categorical predictors in
% the dataset.
%
% Since we are not adjusting for the imbalance of the classes, the
% prediction performnace is not good - only 18 out of 32 people who died
% from hepatitis got identifed, which is just 56% success rate. 

ncat = [2:13,19]; % specify the indices of categorical features
a = fitensemble(T{:,2:end},T.die_or_live,'GentleBoost',150,t,...
  'PredictorNames',T.Properties.VariableNames(2:end),...
  'CategoricalPredictors',ncat,'LearnRate',0.1,'kfold',5);

figure(3);
plot(kfoldLoss(a,'mode','cumulative','lossfun','exponential'));
xlabel('Number of trees');
ylabel('Cross-validated exponential loss');

[Yfit,Sfit] = kfoldPredict(a); 
c = confusionmat(T.die_or_live,Yfit,'order',ClassNames);
CM = table(c(:,1),c(:,2),'VariableNames',ClassNames,'RowNames',ClassNames);
fprintf('Comfusion Matrix\n');
disp(CM)

%% Classification without Cost Adjustment
% This time, the classifier is modified to account for the cost of
% misdiagnosis. If we believe that predicting the patient lives, but the 
% patient dies (false negative) is worse than predicting the patient dies,
% but the patient lives (false negatives), we can adjust the cost of
% misclassification. Let's say the cost is 5x higher for 14 people who
% were misdiagnozed and died than 11 people who were misdiagnozed and 
% survived. 
%
% The result shows some improvement.

cost.ClassNames = ClassNames;
cost.ClassificationCosts = [0 8; 1 0];

aC = fitensemble(T{:,2:end},T.die_or_live,'GentleBoost',150,t,...
  'PredictorNames',T.Properties.VariableNames(2:end),...
  'CategoricalPredictors',ncat,'LearnRate',0.1,'kfold',5,...
  'cost',cost);
[YfitC,SfitC] = kfoldPredict(aC);
c = confusionmat(T.die_or_live,YfitC,'order',ClassNames);
CM = table(c(:,1),c(:,2),'VariableNames',ClassNames,'RowNames',ClassNames);
fprintf('Comfusion Matrix\n');
disp(CM)
