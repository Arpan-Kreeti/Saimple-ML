defmodule NeuralNet do
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
    xor_training_examples = example_gen(&__MODULE__.xor_gate/2)

    [
      {xor_training_examples, "XOR"}
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

    # Train to find correct theta values
    theta =
      1..@iter
      |> Enum.reduce(theta, fn _, weights ->
        calc_iter(training_examples, weights)
      end)

    IO.puts("Trained weights = #{inspect(theta)}")

    # Test model
    test(theta)
  end

  def calc_iter(examples, weights) do
    examples
    |> Enum.reduce(weights, fn {inp1, inp2, output}, [theta11, theta12, theta21] ->
      # Hidden Layer

      # First Activation

      [w1, w2, w3] = theta11

      z11 = w1 * 1 + w2 * inp1 + w3 * inp2

      a11 = sigmoid(z11)

      # Second Activation

      [w1, w2, w3] = theta12

      z12 = w1 * 1 + w2 * inp1 + w3 * inp2

      a12 = sigmoid(z12)

      # Output Layer

      [w1, w2, w3] = theta21

      z21 = w1 * 1 + w2 * a11 + w3 * a12

      h = sigmoid(z21)

      # Backpropagation
      error = output - h
      d_predicted_output = error * sigmoid_grad(h)

      error_hidden_layer = [
        d_predicted_output * w1,
        d_predicted_output * w2,
        d_predicted_output * w3
      ]

      [e1, e2, e3] = error_hidden_layer

      d_hidden_layer = [e1 * sigmoid_grad(1), e2 * sigmoid_grad(a11), e3 * sigmoid_grad(a12)]

      # Updating Weights and Biases

      w1 = w1 + 1 * d_predicted_output * @alpha
      w2 = w2 + a11 * d_predicted_output * @alpha
      w3 = w3 + a12 * d_predicted_output * @alpha

      output_weights = [w1, w2, w3]

      # output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr

      [w1, w2, w3] = theta11

      [dh1, dh2, dh3] = d_hidden_layer

      w1 = w1 + 1 * dh1 * @alpha

      w2 = w2 + inp1 * dh2 * @alpha

      w3 = w3 + inp2 * dh3 * @alpha

      hidden_weight_1 = [w1, w2, w3]



      [w1, w2, w3] = theta12

      [dh1, dh2, dh3] = d_hidden_layer

      w1 = w1 + 1 * dh1 * @alpha

      w2 = w2 + inp1 * dh2 * @alpha

      w3 = w3 + inp2 * dh3 * @alpha

      hidden_weight_2 = [w1, w2, w3]

      # hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

      [hidden_weight_1, hidden_weight_2, output_weights]
    end)
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
