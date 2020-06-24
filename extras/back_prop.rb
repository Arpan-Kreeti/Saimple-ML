require 'pry'
# Range of initial theta value (-INIT_EPSILON, +INIT_EPSILON)
INIT_EPSILON = 0.12
# Learning Rate
ALPHA = 0.001
# Criteria for stopping Gradient descent, target value for cost function
EPSILON = 0.1
# Max iterations for gradient descent
MAX_ITER = 5000

def start 

  theta = [[theta_init(3), theta_init(3)], [theta_init(3)]]

  examples = example_gen(:xnor)

  activations = []
  
  theta1_grad = [0, 0, 0]

  theta2_grad = [0, 0, 0]

  examples.each do |inp_1, inp_2, output|
    result = forward_prop(inp_1, inp_2, output, theta) 
    theta1_grad = theta1_grad.zip(result[:theta1_grad]).map { |val1, val2| val1 + val2}
    theta2_grad = theta2_grad.zip(result[:theta2_grad]).map { |val1, val2| val1 + val2}
    activations << result
  end

  m = activations.length

  theta1_grad = theta1_grad.map { |val| (1.0/m) * val}
  theta2_grad = theta2_grad.map { |val| (1.0/m) * val}

  cost = calc_cost(activations)
end

# Cost function: -1/m * ∑(1 to m) [y*log(h) + (1-y)log(1-h)]
def calc_cost(activations)
  # Number of inputs
  m = activations.length 

  sum = activations.inject(0) do |cost, activation| 

    prediction = activation[:prediction]
    expected_output = activation[:output]

    cost + (expected_output * Math.log(prediction)) + ((1 - expected_output) * Math.log(1 - prediction))
  end

  (-1.0 / m) * sum
end


def example_gen(func, number_of_examples = 1000)
  (1..number_of_examples).map do 

    rand_1 = rand(0..1)
    rand_2 = rand(0..1)

    output = send(func, rand_1, rand_2)

    [rand_1, rand_2, output]
  end
end

def xnor(inp_1, inp_2)
  inp_1 == inp_2 ? 1 : 0
end

def forward_prop(input1, input2, expected_output, theta)
  # Add bias to inputs
  a1 = [1, input1, input2]

  # Destructure activations for 2nd layer
  z21_a21, z22_a22 = calc_activation(a1, theta[0])

  z21, a21 = z21_a21

  z22, a22 = z22_a22

  # Add bias to 2nd layer(hidden layer) activation units
  a2 = [1, a21, a22]

  # Destructure activations for 3nd layer (Ouput prediction)
  # Since we have only one output in last layer, we use splat operator like z31_a31, *
  z31_a31, * = calc_activation(a2, theta[1])

  z31, a31 = z31_a31

  # ERROR CALCULATION

    # Error in last layer output layer:

    # Might require mod value
    delta3 = a31 - expected_output

    # -----------------------------------------

    # Error in 2nd layer/hidden layer: (theta2 * delta3) * sigmoidGrad(z2)

    tmp = theta[1][0].map {|theta_val| theta_val * delta3}

    z2 = [1, z21, z22] # Add bias so we can multiply (since theta2 has 3 vals)

    delta2 = tmp.zip(z2).map { |tmp_val, z2_val| tmp_val * sigmoid_grad(z2_val) }

    #delta2 = delta2.drop(1) # Drop the first value, since we don't need to take into account error in bias

    # -----------------------------------------

    # grad = delta * a

    theta2_grad = a2.map { |val| val * delta3 }

    theta1_grad = a1.zip(delta2).map { |val1, val2| val1 * val2 }

  {inputs: a1, hidden_layer: a2, prediction: a31, output: expected_output, theta1_grad: theta1_grad, theta2_grad: theta2_grad}
end

def calc_activation(input, theta)
  theta.map do |curr_theta|
    z = curr_theta.zip(input).reduce(0) do |acc, nums|
      inp, theta_val = nums
      acc + (inp * theta_val)
    end
    [z, sigmoid(z)]
  end
end

def transform(prediction)
  prediction < 0.5 ? 0 : 1
end

def sigmoid(value)
  1.0 / (1.0 + Math.exp(-value))
end

# g(u) .∗ (1−g(u))
def sigmoid_grad(value)
  g = sigmoid(value)
  g * (1 - g)
end


def theta_init(num)
  max = 1
  min = 0

  (1..num).map do 
    # Get random float between 0 and 1
    tmp = rand() * (max-min) + min 
    # Gives us a random value between [-INIT_EPSILON, +INIT_EPSILON]
    tmp * (2 * INIT_EPSILON) - INIT_EPSILON 
  end
end

# TODO IMPLEMENT GRADIENT DESCENT

# def gradient_descent(cost_func, training_set, m, theta1, theta2, iter)
#   if(iter < MAX_ITER) 
    
#     {cost, grad1, grad2} = cost_func.(m, training_set, theta1, theta2)

#     if(rem(iter, 100) == 0) do
#       IO.puts("-- ITERATION: #{iter} --")
#       IO.puts("Value of cost function = #{cost}")
#       IO.puts("Grad1 = #{grad1}, Grad2 = #{grad2}")
#       IO.puts("Theta1: #{inspect(theta1)}")
#       IO.puts("")
#       IO.puts("Theta2: #{inspect(theta2)}")
#       IO.puts("")
#     end

#     if(cost < @epsilon) do
#       {theta1, theta2}
#     else
#       tmp0 = Enum.map(theta1, fn x -> Enum.map(x, fn y -> y - @alpha * grad1 end) end)
#       tmp1 = Enum.map(theta2, fn x -> Enum.map(x, fn y -> y - @alpha * grad2 end) end)

#       descent(cost_func, training_set, m, tmp0, tmp1, iter + 1)
#     end
#   else
#     {theta1, theta2}
#   end
# end


start()
