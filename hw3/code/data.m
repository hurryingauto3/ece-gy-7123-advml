% Load the data from data1.mat
load('data1.mat');

% List variables in the workspace
whos;

% Check that TrainingX, TrainingY, TestX, TestY exist
if ~exist('TrainingX','var') || ~exist('TrainingY','var') ...
        || ~exist('TestX','var') || ~exist('TestY','var')
    error('Expected variables (TrainingX, TrainingY, TestX, TestY) not found.');
end

% Basic info on training data
disp('--- Training Data ---');
fprintf('TrainingX: %d x %d\n', size(TrainingX,1), size(TrainingX,2));
fprintf('TrainingY: %d x %d\n', size(TrainingY,1), size(TrainingY,2));

% Repeat for test data
disp('--- Test Data ---');
fprintf('TestX: %d x %d\n', size(TestX,1), size(TestX,2));
fprintf('TestY: %d x %d\n', size(TestY,1), size(TestY,2));

disp('---Unique labels in Training/TestY---');
disp(unique(TestY));


% Perform PCA on TrainingX to reduce to 2 dimensions
[coeff, score, ~, ~, explained, mu] = pca(TrainingX);

% Display the percentage of variance explained by the first two components
fprintf('Variance explained by PC1: %.2f%%\n', explained(1));
fprintf('Variance explained by PC2: %.2f%%\n', explained(2));

% Scatter plot of the training data in the PCA space
figure;
gscatter(score(:,1), score(:,2), TrainingY);
title('PCA of Training Data');
xlabel('Principal Component 1');
ylabel('Principal Component 2');

% Optionally, project the TestX data onto the same PCA space
TestX_centered = bsxfun(@minus, TestX, mu);
scoreTest = TestX_centered * coeff;

% Scatter plot of the test data in the PCA space (if desired)
figure;
gscatter(scoreTest(:,1), scoreTest(:,2), TestY);
title('PCA of Test Data');
xlabel('Principal Component 1');
ylabel('Principal Component 2');