# Linear regression with multiple variables
In this project we will implement linear regression with multiple variables to predict the prices of houses. The project is an optional exercise from the ["Machine Learning" course from Andrew Ng](https://www.coursera.org/learn/machine-learning/).

The task is described as follows:
Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to
first collect information on recent houses sold and make a model of housing prices.
The file `ex1data2.txt` contains a training set of housing prices in Portland, Oregon. The first column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price of the house.

The implementation was done using [GNU Octave](https://www.gnu.org/software/octave/). The start point is the `ex1_multi.m` script and other functions are implemented in separate `*.m` files. 

## Feature Normalization
We will start by loading and displaying some values from this dataset.
```matlab
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
```
which produces:

![load data](https://i.imgur.com/sqiYea4.png)

By looking at the values, we note that house sizes are about 1000 times the number of bedrooms. When features differ by orders of magnitude, first performing feature scaling can make gradient descent converge much more quickly.
In order to do that we will:
* Subtract the mean value of each feature from the dataset.
* After subtracting the mean, additionally scale (divide) the feature values by their respective standard deviations.

```matlab
function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));

  mu = mean(X)
  sigma = std(X)

  for iter = 1:size(X,2)
    X_norm(:,iter)=(X(:,iter)-mu(iter))/sigma(iter);
  end
end
```

To take into account the intercept term (θ0), we add an additional first column to X and set it to all ones. This allows
us to treat θ0 as simply another ‘feature’.
The first 10 examples of the modified X matrix look like this:
![normalized features](https://i.imgur.com/0pjdtPk.png)

## Gradient Descent
