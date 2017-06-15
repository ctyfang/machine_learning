function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% mean(X) returns a row vector, where each element in the mean of a col in X  
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

% Standardization is the process of mapping your features to resemble a
% normal(gaussian) distribution
sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);


% ============================================================

end
