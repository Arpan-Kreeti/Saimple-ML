defmodule NeuralNet do
  import Matrix
  # Number of iterations
  @iter 1000
  # Range for initail theta (-init_epsilon, +init_epsilon)
  @init_epsilon 0.12
  # Number of training examples to generate
  @m 3000
  # Learning rate for gradient descent
  @alpha 0.1

  def run do
    # Generate Examples
    xnor_training_examples = [{0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}]

    [
      {xnor_training_examples, "XNOR"}
    ]
    |> Enum.each(fn {training_examples, gate} ->
      IO.puts("")
      IO.puts("==== FOR #{gate} gate ====")
      start(training_examples)
    end)
  end

  def start(training_examples) do
    # Initialize Bias
    [hidden_bias_1] = theta_init(1)

    [hidden_bias_2] = theta_init(1)

    [output_bias] = theta_init(1)

    bias = [hidden_bias_1, hidden_bias_2, output_bias]

    # Initial theta value
    weights = [
      [0.88239638, 0.43029568],
      [0.46651687, 0.05443901],
      [0.20009773, 0.21061059],
      [0.60213225, 0.53527186, 0.39512659]
    ]

    # require IEx
    # IEx.pry()

    # theta = [[-30, 20, 20], [10, -20, -20], [-10, 20, 20]]

    training_set =
      training_examples
      |> Enum.reduce([], fn {inp1, inp2, _output}, acc ->
        acc ++ [[inp1, inp2]]
      end)

    # Train to find correct theta values
    theta =
      1..@iter
      |> Enum.reduce(weights, fn _, weights ->
        # Destructure weights (FORWARD PROP IS CORRECT)
        [theta11, theta12, theta2, bias] = weights

        [hidden_bias_1, hidden_bias_2, output_bias] = bias

        theta2 = [theta2]

        theta1 = [theta11, theta12]

        z1 =
          mult(training_set, Matrix.transpose(theta1))
          |> Enum.map(fn [z11, z12] -> [z11 + hidden_bias_1, z12 + hidden_bias_2] end)

        a1 =
          Enum.map(z1, fn [val1, val2] ->
            [sigmoid(val1), sigmoid(val2)]
          end)

        z2 = mult(a1, transpose(theta2)) |> Enum.map(fn [z2] -> [z2 + output_bias] end)

        a2 =
          Enum.map(z2, fn [val] ->
            [sigmoid(val)]
          end)

        # Backpropagation (PROBLEM WITH BIAS)

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

        # Size = (1000 * 2)
        error_hidden_layer = mult(d_predicted_output, theta2)

        derivative_a1 =
          Enum.map(a1, fn [val1, val2] ->
            [sigmoid_grad(val1), sigmoid_grad(val2)]
          end)

        # Size  =  1000 * 2
        # error_hidden_layer * sigmoid_derivative(hidden_layer_output) # (1000 * 2) .* (1000 * 2)
        d_hidden_layer = emult(error_hidden_layer, derivative_a1)

        # d_predicted_output = 1000 * 3

        # should be 1 * 3
        # 1 * 1000
        # 2 * 1000
        [[dw1], [dw2]] =
          transpose(a1)
          # (2*1000) * (1000*1) = (2*1)
          |> mult(d_predicted_output)
          |> scale(@alpha)

        [[w1, w2]] = theta2

        output_bias =
          output_bias + Enum.reduce(d_predicted_output, 0, fn [e], sum -> sum + e end) * @alpha

        theta2 = [w1 + dw1, w2 + dw2]

        # # =====

        # (2 *1000)
        [[dw11, dw12], [dw21, dw22]] =
          transpose(training_set)
          # (2 *1000) * (1000 * 2) = (2 *2)
          |> mult(d_hidden_layer)
          |> scale(@alpha)

        [w11, w12] = theta11
        [w21, w22] = theta12

        hidden_bias_1 =
          hidden_bias_1 +
            Enum.reduce(d_hidden_layer, 0, fn [e, _], sum -> sum + e end) * @alpha

        hidden_bias_2 =
          hidden_bias_2 +
            Enum.reduce(d_hidden_layer, 0, fn [_, e], sum -> sum + e end) * @alpha

        # hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

        theta11 = [w11 + dw11, w12 + dw12]

        theta12 = [w21 + dw21, w22 + dw22]

        # require IEx
        # IEx.pry()

        # IO.inspect([theta11, theta12, theta2])

        bias = [hidden_bias_1, hidden_bias_2, output_bias]

        [theta11, theta12, theta2, bias]
      end)

    IO.puts("Trained weights = #{inspect(theta)}")

    # Test model
    test(theta)
  end

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

  # [[20, 20], [-20, -20], [20, 20], [-30, 10, -10]]
  def test(weights) do
    training_set = [[0, 0], [0, 1], [1, 0], [1, 1]]

    [theta11, theta12, theta2, bias] = weights

    [hidden_bias_1, hidden_bias_2, output_bias] = bias

    theta2 = [theta2]

    theta1 = [theta11, theta12]

    z1 =
      mult(training_set, Matrix.transpose(theta1))
      |> Enum.map(fn [z11, z12] -> [z11 + hidden_bias_1, z12 + hidden_bias_2] end)

    a1 =
      Enum.map(z1, fn [val1, val2] ->
        [sigmoid(val1), sigmoid(val2)]
      end)

    z2 = mult(a1, transpose(theta2)) |> Enum.map(fn [z2] -> [z2 + output_bias] end)

    a2 =
      Enum.map(z2, fn [val] ->
        [sigmoid(val)]
      end)

    training_set
    |> Enum.with_index()
    |> Enum.each(fn {[inp1, inp2], index} ->
      [prediction] = Enum.at(a2, index)
      # prediction = if prediction < 0.5, do: 0, else: 1
      IO.puts("For input #{inp1}, #{inp2} , PREDICTION = #{prediction}")
    end)
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
