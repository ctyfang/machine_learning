function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Try different SVM Parameters here
%C_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300];
%sig_array = C_array;

%accuracy = 0;

%for i = 1:numel(C_array)
%  
%  for j = 1:numel(sig_array)
%    
%    model = svmTrain(X, y, C_array(i), @(x1, x2) gaussianKernel(x1, x2, sig_array(j)));
%    pred = svmPredict(model, Xval);
%    
%    curr_acc = measure_acc(pred, yval);
%    
%    if(curr_acc > accuracy)
%      index1 = i;
%      index2 = j;
%      accuracy = curr_acc;
%    endif
    
%  endfor
%endfor

%C = C_array(index1);
%sigma = sig_array(index2);
%fprintf('At highest accuracy for CV data, C=%i, sig=%i',C_array(index1),sig_array(index2));

% =========================================================================

end
