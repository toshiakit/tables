%% Fisher Iris Dataset Classification
% Experiment with new Table and Category data types in MATLAB R2013b
% release using Fisher Irirs dataset.
%
%% Dataset
% Features in the dataset are the measurements of iris flowers:
% 
% * Sepal length in cm
% * Sepal width in cm
% * Petal length in cm
% * Petal width in cm
%
% Classes in the dataset identify which type of iris each sample was.
%
% * Setosa
% * Versicolor
% * Virginica
%
% The goal is to predict the type of iris based on the measurements. 

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

%% Plot the dataset to visualize its content - Take 1
% There are 4 features and we can pick two features at a time to plot them.
% Let's start with sepal length vs. sepal width. There is a clear boundary
% between Setosa and other classes, but Versicolor and Virginica are not
% well separated. 

figure(1)
gscatter(T.SL, T.SW, T.class);
xlabel('Sepal length (cm)');
ylabel('Sepal width (cm)');
set(legend,'location','best');

%% Plot the dataset to visualize its content - Take 2
% This time we use petal length vs. petal width. All three classes are now
% much more clearly separated. 

figure(2)
gscatter(T.PL, T.PW, T.class);
xlabel('Petal length (cm)');
ylabel('Petal width (cm)');
set(legend,'location','best');

%% Add a new sample
% To try K-nearest neighbor classifier, we will add a new sample for
% predicting its class. 

new = [3,5,4,2];
hold on
line(new(3),new(4),'marker','x','color','k',...
   'markersize',10,'linewidth',2)
text(new(3)-1.15,new(4),'new sample')
hold off

%% Classify a new sample with k nearest neighbor (KNN) method
% We will fit a model using the default parameters and see what class it
% predicts for the new sample. Prediction Score indicates how confident the
% prediction is. 

% Fit a KNN classifier
cl = ClassificationKNN.fit(T{:,1:4},cellstr(T.class),...
    'PredictorNames',T.Properties.VariableNames(1:4),...
    'ResponseName',T.Properties.VariableNames{5}...
    );
% Inspect the model properties 
disp(cl)
% Make a prediction for the new sample
[pred, score] = predict(cl,new);
fprintf('Predicted class of the new sample: "%s"\n', char(pred));
fprintf('Prediction Score: %.2f%%\n\n', score(strcmp(char(pred),...
    cellstr(unique(T.class))))*100);

clearvars pred score

%% Quality of KNN classifier
% The quality of KNN classifier depends on the choice of parameter k, which
% specifies the number of nearest neighbors to use in computation. Smaller 
% k is sensitive to noise in the data.
%
% The default is k = 1, which means it assigns the class of the single 
% nearest neighbor. Since we used the whole dataset to fit the classifier,
% it is going to predict the class of the existing data perfectly, but this
% is not very useful in practice. 

rloss = resubLoss(cl);
fprintf('Resubstitution loss at k=1: %.2f%%\n\n', rloss*100);

clearvars rloss

%% Simulate the realistic performnace with cross validation
% Cross validation enables evaluation of the classifer by withholding a
% portion of the data during the training, and test the classifier against
% the withheld data. The average error rate will give us more realisitc
% idea about the performnace of the classifier. 

cv = crossval(cl);
kloss = kfoldLoss(cv);
fprintf('Avg cross validation loss at k=1: %.2f%%\n\n', kloss*100);

clearvars cv kloss

%% Find the optimal k for KNN classifier
% We can try various values for k and select the value that gives us the 
% lowest error rate. 

k = 1:10;
losses = zeros(1,10);
for i = k
    cl.NumNeighbors = i;
    losses(i) = resubLoss(cl);
end

figure(3)
plot(k,losses)
xlabel('k')
ylabel('Resub loss')

% exclude k=1 from selection
losses(1)=max(losses);
% find the first lowest value and get its index
[~, optimal]=min(losses);
cl.NumNeighbors = optimal;
cv = crossval(cl);
kloss = kfoldLoss(cv);
fprintf('Optimal value of k = %d\n', optimal);
fprintf('Resubstitution loss: %.2f%%\n', losses(optimal)*100);
fprintf('Avg cross validation loss: %.2f%%\n\n', kloss*100);

clearvars i kloss

%% Using cross validation for k selection
% We can also try k-fold loss for k selection. The result changes each time
% you run the code because of random data sampling affects the fit of the
% classifier. 

cvlosses = zeros(1,10);
for i = k
    cl.NumNeighbors = i;
    cv = crossval(cl);
    cvlosses(i) = kfoldLoss(cv);
end

figure(4)
plot(k,cvlosses)
xlabel('k')
ylabel('Avg cross validation loss')

% exclude k=1 from selection
losses(1)=max(cvlosses);
% find the first lowest value and get its index
[~, cvoptimal]=min(cvlosses);
fprintf('Optimal value of k = %d\n', cvoptimal);
fprintf('Resubstitution loss: %.2f%%\n', losses(cvoptimal)*100 );
fprintf('Avg cross validatoin loss: %.2f%%\n\n', cvlosses(cvoptimal)*100);

clearvars i k

%% Confusion Matrix
% Confusion Matrix allows comparison of prediction against the ground truth
% and serves as the basis to compute the rates of true positives, false
% positiives, true negatives, and false negatives. Ground truth is mapped
% to the rows and prediction is mapped to the columns. The diagonal
% represents the true positivies. 
%
% This analysis shows that all misclassifications are between versicolor 
% and virginica. 

labels = cellstr(unique(T.class));
fprintf('Confusion Matrix for k = %d\n', optimal);
cl.NumNeighbors = optimal;
pred = resubPredict(cl);
C = confusionmat(cellstr(T.class),pred);
C1 = table(C(:,1),C(:,2),C(:,3),'VariableNames',labels,'RowNames',labels);
disp(C1)

fprintf('Confusion Matrix for k = %d\n', cvoptimal);
cl.NumNeighbors = cvoptimal;
pred = resubPredict(cl);
C = confusionmat(cellstr(T.class),pred);
C2 = table(C(:,1),C(:,2),C(:,3),'VariableNames',labels,'RowNames',labels);
disp(C2)

clearvars labels pred c C1 C2

%% Plot ROC Performance Curves
% Compare the ROC curves for the two k values. The closer the curves are to
% the left top corner of the plot, the better. 

cl.NumNeighbors = optimal;
[~,score] = resubPredict(cl);
[X1,Y1,~,~] = perfcurve(T.class,score(:,3),'virginica');
cl.NumNeighbors = cvoptimal;
[~,cvscore] = resubPredict(cl);
[X2,Y2,~,~] = perfcurve(T.class,cvscore(:,3),'virginica');

figure(5)
plot(X1,Y1)
hold on
plot(X2,Y2,'r')
hold off
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for KNN classification for "virginica"')
legend(sprintf('k=%d',optimal),sprintf('k=%d',cvoptimal),...
    'Location','best')

clearvars i X1 X2 Y1 Y2

