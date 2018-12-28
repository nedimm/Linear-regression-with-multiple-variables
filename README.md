# Linear regression with multiple variables
In this project we will implement linear regression with multiple variables to predict the prices of houses. The project is an optional exercise from the ["Machine Learning" course from Andrew Ng](https://www.coursera.org/learn/machine-learning/).

The task is described as follows:
Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to
first collect information on recent houses sold and make a model of housing prices.
The file `ex1data2.txt` contains a training set of housing prices in Portland, Oregon. The first column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price of the house.

We will use two approaches to solve this problem:
* Gradient Descent and
* Normal Equations

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

When normalizing the features, it is important to store the values used for normalization - the mean value and the standard deviation used for the computations. After learning the parameters from the model, we often want to predict the prices of houses we have not seen before. Given a new x value (living room area and number of bedrooms), we must first normalize x using the mean and standard deviation that we had previously computed from the training set.

## Gradient Descent

One way to implement linear regression with multiple variables is to use the gradient descent approach.
A possible implementation of the cost function could be:

```matlab
function J = computeCostMulti(X, y, theta)
    m = length(y); % number of training examples
    J = 1/(2*m)*sum(((X*theta)-y).^2);
end
```

In the multivariate case, the cost function can also be written in the following vectorized form:
![vectorized form](https://i.imgur.com/IFVECwD.png)
where 
![X, y](https://i.imgur.com/tADA9vf.png)

The vectorized version is efficient when you’re working with numerical computing tools like Octave/MATLAB.

The Gradient Descent is then implemented as follows:
```matlab
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);
    for iter = 1:num_iters
        delta = (1/m*(X*theta-y)' * X)';
        theta = theta-alpha*delta;
        J_history(iter) = computeCostMulti(X, y, theta);
    end
end
```
Running this code produces:
![](https://i.imgur.com/64vPIRd.png)

In the figure below we see a plot of the convergence graph:
![asdf](https://i.imgur.com/VCM54Wi.png)

Now we can estimate the price of a 1650 sq-ft, 3 bedroom house:
```matlab
d = [1650 3];
d = (d - mu) ./ sigma;
d = [ones(1, 1) d];
price = d * theta;
```
![](https://i.imgur.com/Br2qg15.png)

# Normal Equations

The closed-form solution to linear regression is
![normal equations](https://i.imgur.com/KduFQMS.png)
Using this formula does not require any feature scaling, and you will get an exact solution in one calculation: there is no "loop until convergence" like in gradient descent.

```matlab
function [theta] = normalEqn(X, y)
    theta = pinv(X' * X) * X' * y;
end
```
![](https://i.imgur.com/9MXLcWZ.png)

# Conclusion

In this exercise we used two different approaches for solving the problem of house price prediction, namely the gradient descent and normal equations. The result was roughly the same. We could use the normal equations approach since the number of features was very low. In the case of high number of features (< 10 000) we should use the gradient descent in which case we have to experiment with the learning rate.