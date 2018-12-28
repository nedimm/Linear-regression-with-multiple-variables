%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables

%% ================ Part 1: Feature Normalization ================

fprintf('Loading data ...\n\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('\tFirst 10 examples from the dataset: \n');
fprintf('\t x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

fprintf('\n\tFirst 10 examples from the normalized features: \n');
fprintf('\t x = [%.0f %.2f %.2f] \n', [X(1:10,:)]');

printf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('\tTheta computed from gradient descent: \n');
fprintf('\t %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
d = [1650 3];
d = (d - mu) ./ sigma;
d = [ones(1, 1) d];
price = d * theta;

fprintf(['\tPredicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n\t $%f\n\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n\n');

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('\tTheta computed from the normal equations: \n');
fprintf('\t %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
d = [1 1650 3];
price = d * theta;

fprintf(['\tPredicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n\t $%f\n\n'], price);
