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
        [[dw1], [dw2], [dw3]] =
          transpose(a1)
          |> mult(d_predicted_output)
          |> scale(@alpha)

        [[w1, w2, w3]] = theta2

        theta2 = [w1 + dw1, w2 + dw2, w3 + dw3]

        # # =====

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

  def test(weights) do
    training_set = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

    [theta11, theta12, theta2] = weights

    theta2 = [theta2]

    theta1 = [theta11, theta12]

    z1 = mult(training_set, Matrix.transpose(theta1))

    a1 =
      Enum.map(z1, fn [val1, val2] ->
        [1, sigmoid(val1), sigmoid(val2)]
      end)

    z2 = mult(a1, transpose(theta2))

    a2 =
      Enum.map(z2, fn [val] ->
        [sigmoid(val)]
      end)

    training_set
    |> Enum.with_index()
    |> Enum.each(fn {[_, inp1, inp2], index} ->
      [prediction] = Enum.at(a2, index)
      prediction = if prediction < 0.5, do: 0, else: 1
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
