defmodule NeuralNet do
  import Matrix
  # Number of iterations
  @iter 1000
  # Range for initail theta (-init_epsilon, +init_epsilon)
  @init_epsilon 0.12
  # Number of training examples to generate
  @m 1000
  # Learning rate for gradient descent
  @alpha 0.1

  def run do
    # Generate Examples
    xnor_training_examples = example_gen(&__MODULE__.xnor_gate/2)

    [
      {xnor_training_examples, "XOR"}
    ]
    |> Enum.each(fn {training_examples, gate} ->
      IO.puts("")
      IO.puts("==== FOR #{gate} gate ====")
      start(training_examples)
    end)
  end

  def start(training_examples) do
    # Initial theta value
    theta = [theta_init(3), theta_init(3), theta_init(3)]

    # theta = [[-30, 20, 20], [10, -20, -20], [-10, 20, 20]]

    training_set =
      training_examples
      |> Enum.reduce([], fn {inp1, inp2, _output}, acc ->
        acc ++ [[1, inp1, inp2]]
      end)

    # Train to find correct theta values
    theta =
      1..@iter
      |> Enum.reduce(theta, fn _, weights ->
        # Destructure weights
        [theta11, theta12, theta2] = weights

        theta2 = [theta2]

        # CALCULATE HIDDEN LAYER ACTIVATION

        # Destructure hidden layer weights
        # size = (2 * 3)
        theta1 = [theta11, theta12]

        # size = (1000 * 3) * (3 * 2) = (1000 * 2)
        z1 = mult(training_set, Matrix.transpose(theta1))

        # Compute sigmoid, and add bias
        a1 =
          Enum.map(z1, fn [val1, val2] ->
            [1, sigmoid(val1), sigmoid(val2)]
          end)

        # CALCULATE OUTPUT LAYER ACTIVATION

        # size = (1000 * 2)
        z2 = mult(a1, transpose(theta2))

        a2 =
          Enum.map(z2, fn [val] ->
            [sigmoid(val)]
          end)

        # Backpropagation

        outputs =
          Enum.map(training_examples, fn {_inp1, _inp2, output} ->
            [output]
          end)

        error = sub(outputs, a2)

        derivative_a2 =
          Enum.map(a2, fn [val] ->
            [sigmoid_grad(val)]
          end)

        # Size = (1000 * 1)
        d_predicted_output = emult(error, derivative_a2)

        # Hidden layer

        # Size = (1000 * 3)
        error_hidden_layer = mult(d_predicted_output, theta2)

        derivative_a1 =
          Enum.map(a1, fn [val1, val2, val3] ->
            [sigmoid_grad(val1), sigmoid_grad(val2), sigmoid_grad(val3)]
          end)

        # Size  =  1000 * 3
        d_hidden_layer = emult(error_hidden_layer, derivative_a1)

        # d_predicted_output = 1000 * 3

        # should be 1 * 3
        # 1 * 1000
        [[dw]] =
          transpose(a2)
          |> mult(d_predicted_output)
          |> scale(@alpha)

        [[w1, w2, w3]] = theta2

        theta2 = [w1 + dw, w2 + dw, w3 + dw]

        # =====

        [_, [dw11, dw12, dw13], [dw21, dw22, dw23]] =
          transpose(a1)
          |> mult(d_hidden_layer)
          |> scale(@alpha)

        [w11, w12, w13] = theta11
        [w21, w22, w23] = theta12

        theta11 = [w11 + dw11, w12 + dw12, w13 + dw13]

        theta12 = [w21 + dw21, w22 + dw22, w23 + dw23]

        [theta11, theta12, theta2]
      end)

    IO.puts("Trained weights = #{inspect(theta)}")

    # Test model
    # test(theta)
  end

  # def calc_iter(examples, weights) do
  #   examples
  #   |> Enum.reduce(weights, fn {inp1, inp2, output}, [theta11, theta12, theta21] ->
  #     # Hidden Layer

  #     # First Activation

  #     [w1, w2, w3] = theta11

  #     z11 = w1 * 1 + w2 * inp1 + w3 * inp2

  #     a11 = sigmoid(z11)

  #     # Second Activation

  #     [w1, w2, w3] = theta12

  #     z12 = w1 * 1 + w2 * inp1 + w3 * inp2

  #     a12 = sigmoid(z12)

  #     # Output Layer

  #     [w1, w2, w3] = theta21

  #     z21 = w1 * 1 + w2 * a11 + w3 * a12

  #     h = sigmoid(z21)

  #     # Backpropagation
  #     error = output - h
  #     d_predicted_output = error * sigmoid_grad(h)

  #     error_hidden_layer = [
  #       d_predicted_output * w1,
  #       d_predicted_output * w2,
  #       d_predicted_output * w3
  #     ]

  #     [e1, e2, e3] = error_hidden_layer

  #     d_hidden_layer = [e1 * sigmoid_grad(1), e2 * sigmoid_grad(a11), e3 * sigmoid_grad(a12)]

  #     # Updating Weights and Biases

  #     w1 = w1 + 1 * d_predicted_output * @alpha
  #     w2 = w2 + a11 * d_predicted_output * @alpha
  #     w3 = w3 + a12 * d_predicted_output * @alpha

  #     output_weights = [w1, w2, w3]

  #     # output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr

  #     [w1, w2, w3] = theta11

  #     [dh1, dh2, dh3] = d_hidden_layer

  #     w1 = w1 + 1 * dh1 * @alpha

  #     w2 = w2 + inp1 * dh2 * @alpha

  #     w3 = w3 + inp2 * dh3 * @alpha

  #     hidden_weight_1 = [w1, w2, w3]

  #     [w1, w2, w3] = theta12

  #     [dh1, dh2, dh3] = d_hidden_layer

  #     w1 = w1 + 1 * dh1 * @alpha

  #     w2 = w2 + inp1 * dh2 * @alpha

  #     w3 = w3 + inp2 * dh3 * @alpha

  #     hidden_weight_2 = [w1, w2, w3]

  #     # hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

  #     [hidden_weight_1, hidden_weight_2, output_weights]
  #   end)
  # end

  def sigmoid(x) do
    1 / (1 + :math.exp(-x))
  end

  def sigmoid_grad(value) do
    g = sigmoid(value)
    g * (1 - g)
  end

  def example_gen(func) do
    1..@m
    |> Enum.map(fn _ ->
      rand_1 = rand()
      rand_2 = rand()

      output = func.(rand_1, rand_2)

      {rand_1, rand_2, output}
    end)
  end

  def test(weights) do
    [{0, 0}, {0, 1}, {1, 0}, {1, 1}]
    |> Enum.each(fn {inp1, inp2} ->
      [theta11, theta12, theta21] = weights

      [w1, w2, w3] = theta11

      a11 = sigmoid(w1 * 1 + w2 * inp1 + w3 * inp2)

      # --------------------

      [w1, w2, w3] = theta12

      a12 = sigmoid(w1 * 1 + w2 * inp1 + w3 * inp2)

      # --------------------

      [w1, w2, w3] = theta21

      prediction = sigmoid(w1 * 1 + w2 * a11 + w3 * a12)

      output = if prediction < 0.5, do: 0, else: 1

      IO.puts("For input #{inp1}, #{inp2} , PREDICTION = #{prediction}")
    end)
  end

  def and_gate(inp_1, inp_2) do
    if inp_1 == 1 && inp_2 == 1, do: 1, else: 0
  end

  def nand_gate(inp_1, inp_2) do
    if and_gate(inp_1, inp_2) == 0, do: 1, else: 0
  end

  def or_gate(inp_1, inp_2) do
    if inp_1 == 1 || inp_2 == 1, do: 1, else: 0
  end

  def nor_gate(inp_1, inp_2) do
    if or_gate(inp_1, inp_2) == 0, do: 1, else: 0
  end

  def xor_gate(inp1, inp2) do
    if inp1 == inp2, do: 0, else: 1
  end

  def xnor_gate(inp1, inp2) do
    if xor_gate(inp1, inp2) == 0, do: 1, else: 0
  end

  def rand do
    :rand.uniform(100_000)
    |> rem(2)
  end

  def theta_init(n) do
    Enum.reduce(1..n, [], fn _, acc ->
      rand = :rand.uniform(999) / 1000 * (2 * @init_epsilon) - @init_epsilon
      acc ++ [rand]
    end)
  end
end
