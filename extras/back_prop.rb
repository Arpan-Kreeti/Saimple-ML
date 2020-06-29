require 'pry'
# Range of initial theta value (-INIT_EPSILON, +INIT_EPSILON)
INIT_EPSILON = 0.12
# Learning Rate
ALPHA = 0.01
# Criteria for stopping Gradient descent, target value for cost function
EPSILON = 0.1
# Max iterations for gradient descent
MAX_ITER = 5000

def start(theta, examples)

  activations = []
  
  theta1_grad = [[0, 0, 0], [0, 0, 0]]

  theta2_grad = [0, 0, 0]

  examples.each do |inp_1, inp_2, output|
    result = forward_prop(inp_1, inp_2, output, theta) 

    theta1_grad = theta1_grad.zip(result[:theta1_grad]).map { |val1, val2| 
      val1.zip(val2).map {|v1, v2| v1 + v2}
    }

    theta2_grad = theta2_grad.zip(result[:theta2_grad]).map { |val1, val2| val1 + val2}
    activations << result
  end

  m = activations.length

  theta1_grad = theta1_grad.map { |val|  val.map{ |v| (1.0/m) * v }  }

  theta2_grad = theta2_grad.map { |val| (1.0/m) * val}

  [theta1_grad, theta2_grad]
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

  # Destructure activations for 3rd layer (Ouput prediction)
  # Since we have only one output in last layer, we use splat operator like z31_a31, *
  z31_a31, * = calc_activation(a2, theta[1])

  z31, a31 = z31_a31

  # ERROR CALCULATION

    # Error in last layer output layer:

    # Might require mod value
    delta3 = a31 - expected_output

    # -----------------------------------------

    # Error in 2nd layer/hidden layer: (theta2 * delta3) * sigmoidGrad(z2)

    tmp = theta[0][0].map {|theta_val| theta_val * delta3}

    z2 = [1, z21, z22] # Add bias so we can multiply (since theta2 has 3 vals)

    delta21 = tmp.zip(z2).map { |tmp_val, z2_val| tmp_val * sigmoid_grad(z2_val) }

    
    tmp = theta[0][1].map {|theta_val| theta_val * delta3}

    z2 = [1, z21, z22] # Add bias so we can multiply (since theta2 has 3 vals)

    delta22 = tmp.zip(z2).map { |tmp_val, z2_val| tmp_val * sigmoid_grad(z2_val) }

    delta2 = [delta21, delta22]

    # -----------------------------------------
    # Accumulating Errors
    # Δ = Δ + a * delta

    # For 2nd -> 3rd layer
    theta2_grad = a2.map { |val| val * delta3 }

    # For 1st -> 2nd layer
    tmp1 = a1.zip(delta2[0]).map { |val1, val2| val1 * val2 }

    tmp2 = a1.zip(delta2[1]).map { |val1, val2| val1 * val2 }

    theta1_grad = [tmp1, tmp2]

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

def gradient_descent()

  theta = [ 
            [theta_init(3), theta_init(3)], 
            [theta_init(3)]
          ]

  examples = example_gen(:xnor)

  500.times {

    theta1_grad, theta2_grad = start(theta, examples)

    theta_01 = theta[0][0].zip(theta1_grad[0]).map { |val1, val2|
      val1 - ALPHA * val2
    }

    theta_02 = theta[0][1].zip(theta1_grad[1]).map { |val1, val2|
      val1 - ALPHA * val2
    }

    theta1 = [theta_01, theta_02]

    theta2 = theta[1][0].zip(theta2_grad).map { |val1, val2| 
      val1 - ALPHA * val2
    }

    theta = [theta1, [theta2]]
  }

  p theta
end

def xnor_forward_prop(input1, input2, theta_1, theta_2)

  a1 = [1, input1, input2]

  z21_a21, z22_a22 = calc_activation(a1, theta_1)

  z21, a21 = z21_a21

  z22, a22 = z22_a22

  # Add bias to 2nd layer(hidden layer) activation units
  a2 = [1, a21, a22]

  # Destructure activations for 3rd layer (Ouput prediction)
  # Since we have only one output in last layer, we use splat operator like z31_a31, *
  z31_a31, * = calc_activation(a2, theta_2)

  z31, a31 = z31_a31

  return a31
end

# # Cost function: -1/m * ∑(1 to m) [y*log(h) + (1-y)log(1-h)]
# def calc_cost(input)
#   # Number of inputs
#   m = input.length 

#   sum = input.inject(0) do |x1, x2, y, theta1, theta2| 

#     h =  xnor_forward_prop(x1, x2, theta_1, theta_2)

#     cost + (y * Math.log(h)) + ((1 - y) * Math.log(1 - h))
#   end

#   (-1.0 / m) * sum
# end


gradient_descent()
