function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
 % load('ex4weights.mat');
 % nn_params = [ Theta1(:); Theta2(:) ] % 10285
 % input_layer_size = 400
 % hidden_layer_size = 25
 % num_labels = 10
 % lambda = 0
 
                               
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% Forward Propagation

% ------ LAYER 2 - HIDDEN LAYER --------

% Add bias row input
a1 = [ones(m, 1) X]; % 5000*401

z2 = Theta1 * a1'; % (25*401)*(401*5000)

% Calculate activations for 2nd layer (Hidden layer)
a2 = sigmoid(z2); % (25*5000)

% ------ LAYER 3 - OUTPUT LAYER --------

% Add bias row
a2 = [ones(m, 1) a2'];

z3 = Theta2 * a2';

% Calculate activations for 3rd layer (Output layer)
a3 = sigmoid(z3);

% The original labels (in the variable y) were 1, 2, ..., 10, 
% for the purpose of training a neural network, we need to recode the 
% labels as vectors containing only values 0 or 1.
% So size(y) = 5000 * 1, out vector will have size = 5000 * 10

% Empty matrix of zeros, 10 * 5000
y_matrix = zeros(num_labels, m);

% For each ith training example, set the value of the jth column of the
% matrix. In the jth column set the value of the correct row depending on
% the value of y. 
% So, for jth example if y(j,1) is 4 then set the 4th
% row for the jth column to 1.
for j = 1:m 
   y_matrix(y(j), j) = 1; 
end;

% Calculate the cost function, using the hypothesis we obtained above..
% Here, y_matrix is a matrix of size (5000 * 10), since each output can be 
% calssified in 10 classes unlike earlier for
% logistic regression where y was a vector (5000 * 1). 
% There here we do element wise multiplication with y_matrix.

% The summation terms account for the sum of all classes (1..10) and then
% the sum of values for all input(1..5000).
J = (1/m) * sum( sum( (-y_matrix) .* log(a3) - (1 - y_matrix) .* log(1 - a3))); 

% Regularizing our cost function
% Go through ex4.pdf pg: 6 for formula explaination

% We should not be regularizing the terms that correspond to the bias. 
% For the matrices Theta1 and Theta2, this corresponds to the first
% column of each matrix.

new_Theta1 = Theta1;
new_Theta2 = Theta2;

% Remove 1st columns from theta matrices which correspond to bias terms
new_Theta1(:,1) = [];
new_Theta2(:,1) = [];

tmp1 = sum( sum(new_Theta1 .^ 2));

tmp2 = sum( sum(new_Theta2 .^ 2));

regularization_term = lambda/(2*m) * (tmp1 + tmp2);

J = J + regularization_term;
% -------------------------------------------------------------

% Backpropagation

%delta2 = zeros(num_labels, m); % 10 * 5000
%delta1 = zeros(26, m); % 26 * 5000

delta3_acc = zeros(25, 10);

delta2_acc = zeros(9, 26);

% For each training Example
for t = 1:m
    
    % -- STEP 1: Forward Propagation --
    
    % size(X) = 5000 * 400 (Each row is an input with 400 features)
    % X(1) = 1 * 400
    
    % The first layer is same as our input 
    % So 400 inputs (including bias), a1 is a column vector (400*1)
    a1 = X(t, :)'; 
    
    % Add bias input, a row of 1's,
    % So, now size(a1) = 401 * 1
    a1 = [1; a1]; 

    z2 = Theta1 * a1;  %(25 *401) * (401 *1) = (25*1)

    % Calculate activations for 2nd layer (Hidden layer)
    a2 = sigmoid(z2); % (25 * 1)
    
    % Add bias input, a row of 1's
    % Now, size(a2) = 26 * 1
    a2 = [1; a2];

    z3 = Theta2 * a2; % (10 * 26) * (26* 1) = (10 * 1)

    % Calculate activations for 3rd layer(Output layer)/ Final hypothesis
    a3 = sigmoid(z3);
    
    %-------------------------------------------
    
    % -- STEP 2 -- Calculating delta/errors in output layer
    
    % Calculating error in output layer/ Layer 3
    
    % We use the matrix form of output(y) here
    % size(y) = 10 *5000, each column is a output
    % represented by a vector of size(10) for 10 classes
    
    % Thus y_matrix(:,t) gives us the tth output
    
    % Place the difference in first column of delta matrix
    delta3 = a3 - y_matrix(:,t); % (10 * 1)
    
    % Calculating error in hidden layer/ Layer 2
    % We add a row to z2 for bias
    z2 = [1; z2]; % 26 * 1
    
    % -- STEP 3 -- Calculating delta/errors in hidden layer  
    delta2 = (Theta2' * delta3) .* sigmoidGradient(z2);% (26 *10) * (10 *1).* (26 * 1) = (26 *1 )
    
    
    % -- STEP 4 -- Accumultaing delta/errors in each layer  

    % When accumulating the errors of a layer 
    % we should skip the 1st row in delta since its the error corresponding
    % to the bias input, so we do delta2(1:end)
    delta2 = delta2(2:end);
    
    % Accumultaing the errors in layers 2 and 3 
    % (in input layer there is no error)
    Theta2_grad = Theta2_grad + delta3 * a2'; % (10*1)*(1*26)
	Theta1_grad = Theta1_grad + delta2 * a1'; % (25*1)*(1*401)
end;

% -- STEP 5 -- Finding final gradient

% Final gradient

Theta1_grad = (1/m) * Theta1_grad; % (25*401)

Theta2_grad = (1/m) * Theta2_grad; % (10*26)


% Adding regularization to the obtained gradients

% We should ignore the first column which is for the bias values
% when we apply regularization

% Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m; % for j = 0
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 

% Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m; % for j = 0
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1


% =========================================================================

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
