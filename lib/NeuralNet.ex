defmodule NeuralNet do
  import Matrix
  # Number of iterations
  @iter 100000
  # Learning rate
  @alpha 0.1

  def run do
    # Generate Examples
    xor_training_examples = [{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}]
    xnor_training_examples = [{0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}]

    [
      {xor_training_examples, "XOR"},
      {xnor_training_examples, "XNOR"}
    ]
    |> Enum.each(fn {training_examples, gate} ->
      IO.puts("")
      IO.puts("==== FOR #{gate} gate ====")
      start(training_examples)
    end)
  end

  def start(training_examples) do
    # Initial weights and Bias
    [hidden_bias_1] = theta_init(1)

    [hidden_bias_2] = theta_init(1)

    [output_bias] = theta_init(1)

    bias = [hidden_bias_1, hidden_bias_2, output_bias]

    weights = [theta_init(2), theta_init(2), theta_init(2), bias]

    # Prepare training set
    training_set =
      training_examples
      |> Enum.reduce([], fn {inp1, inp2, _output}, acc ->
        acc ++ [[inp1, inp2]]
      end)

    # Train to find correct weights and bias values
    theta =
      1..@iter
      |> Enum.reduce(weights, fn _, weights ->
        # ==== FORWARD PROPAGATION =====

        # Destructure weights
        [theta11, theta12, theta2, bias] = weights

        # Destructure biases
        [hidden_bias_1, hidden_bias_2, output_bias] = bias

        theta2 = [theta2]

        theta1 = [theta11, theta12]

        # Calculate activation and add bias to each activation
        z1 =
          mult(training_set, theta1)
          |> Enum.map(fn [z11, z12] -> [z11 + hidden_bias_1, z12 + hidden_bias_2] end)

        # Apply sigmoid function to find final hidden layer activation
        a1 =
          Enum.map(z1, fn [val1, val2] ->
            [sigmoid(val1), sigmoid(val2)]
          end)

        # Calculate output activation and add bias to it
        z2 = mult(a1, transpose(theta2)) |> Enum.map(fn [z2] -> [z2 + output_bias] end)

        # Apply sigmoid function to find final activation/predicted output
        a2 =
          Enum.map(z2, fn [val] ->
            [sigmoid(val)]
          end)

        # ==== BACK PROPAGATION =====

        # Get all actual output values
        outputs =
          Enum.map(training_examples, fn {_inp1, _inp2, output} ->
            [output]
          end)

        # Find error in prediction, ERROR = EXPECTED_OUTPUT - PREDICTION
        error = sub(outputs, a2)

        # Find sigmoid_derivative of prediction
        derivative_a2 =
          Enum.map(a2, fn [val] ->
            [sigmoid_grad(val)]
          end)

        # So, ERROR_IN_OUTPUT_LAYER = ERROR .* SIGMOID_DERIVATIVE(PREDICTION)
        d_predicted_output = emult(error, derivative_a2)

        # == Now, calculating error in hidden layer == #

        # ERROR = ERROR_IN_OUTPUT_LAYER * OUTPUT_LAYER_WEIGHTS
        error_hidden_layer = mult(d_predicted_output, theta2)

        # Find sigmoid_derivative of hidden layer activations
        derivative_a1 =
          Enum.map(a1, fn [val1, val2] ->
            [sigmoid_grad(val1), sigmoid_grad(val2)]
          end)

        # So, ERROR_IN_HIDDEN_LAYER = ERROR .* SIGMOID_DERIVATIVE(HIDDEN_LAYER_ACTIVATIONS)
        d_hidden_layer = emult(error_hidden_layer, derivative_a1)

        #  ==== ADJUST WEIGHTS ====

        # OUTPUT LAYER

        # DELTA_WEIGHT_OUTPUT = (PREDICTION * ERROR_IN_OUTPUT_LAYER) * ALPHA
        [[dw1], [dw2]] =
          transpose(a1)
          |> mult(d_predicted_output)
          |> scale(@alpha)

        [[w1, w2]] = theta2

        # Adjust output layer bias
        # OUTPUT_BIAS += SUM_OF_ERRORS_IN_OUTPUT_LAYER_ACROSS_ALL_TRAINING_DATA * ALPHA
        output_bias =
          output_bias + Enum.reduce(d_predicted_output, 0, fn [e], sum -> sum + e end) * @alpha

        # Adjust output layer weights by adding delta
        theta2 = [w1 + dw1, w2 + dw2]

        # HIDDEN LAYER

        # DELTA_WEIGHT_HIDDENT = (INPUT * ERROR_IN_LAYER_LAYER) * ALPHA
        [[dw11, dw12], [dw21, dw22]] =
          transpose(training_set)
          |> mult(d_hidden_layer)
          |> scale(@alpha)

        [w11, w12] = theta11
        [w21, w22] = theta12

        # Adjust hidden layer biases
        # HIDDEN_BIAS += SUM_OF_ERRORS_IN_HIDDEN_LAYER_NODE_ACROSS_ALL_TRAINING_DATA * ALPHA
        hidden_bias_1 =
          hidden_bias_1 +
            Enum.reduce(d_hidden_layer, 0, fn [e, _], sum -> sum + e end) * @alpha

        hidden_bias_2 =
          hidden_bias_2 +
            Enum.reduce(d_hidden_layer, 0, fn [_, e], sum -> sum + e end) * @alpha

        # Adjust hidden layer weights by adding delta
        theta11 = [w11 + dw11, w12 + dw12]

        theta12 = [w21 + dw21, w22 + dw22]

        # ---- BACK PROP END ---- #

        bias = [hidden_bias_1, hidden_bias_2, output_bias]

        [theta11, theta12, theta2, bias]
      end)

    [
      hidden_weight_1,
      hidden_weight_2,
      output_weights,
      bias
    ] = theta

    [hidden_bias_1, hidden_bias_2, output_bias] = bias

    IO.puts("Final hidden weights: #{inspect([hidden_weight_1, hidden_weight_2])}")
    IO.puts("Final hidden bias: #{inspect([hidden_bias_1, hidden_bias_2])}")
    IO.puts("Final output weights: #{inspect(output_weights)}")
    IO.puts("Final output bias: #{inspect(output_bias)}")

    # Test model
    test(theta)
  end

  def sigmoid(x) do
    1 / (1 + :math.exp(-x))
  end

  def sigmoid_grad(value) do
    value * (1 - value)
    # g = sigmoid(value)
    # g * (1 - g)
  end

  def test(weights) do
    training_set = [[0, 0], [0, 1], [1, 0], [1, 1]]

    [theta11, theta12, theta2, bias] = weights

    [hidden_bias_1, hidden_bias_2, output_bias] = bias

    theta2 = [theta2]

    theta1 = [theta11, theta12]

    z1 =
      mult(training_set, theta1)
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
      rounded_up = if prediction < 0.5, do: 0, else: 1
      IO.puts("For input #{inp1}, #{inp2} , PREDICTION = #{prediction} ~ #{rounded_up}")
    end)
  end

  def xor_gate(inp1, inp2) do
    if inp1 == inp2, do: 0, else: 1
  end

  def xnor_gate(inp1, inp2) do
    if xor_gate(inp1, inp2) == 0, do: 1, else: 0
  end

  def theta_init(n) do
    Enum.reduce(1..n, [], fn _, acc ->
      acc ++ [:random.uniform()]
    end)
  end
end
