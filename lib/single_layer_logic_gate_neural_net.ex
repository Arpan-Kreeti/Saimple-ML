defmodule SingleLayerLogicGateNeuralNet do
  @iter 1000           # Number of iterations
  @init_epsilon 0.12   # Range for initail theta (-init_epsilon, +init_epsilon)
  @m 1000              # Number of training examples to generate
  @alpha 0.1           # Learning rate for gradient descent

  def run do
    # Generate Examples
    and_training_examples = example_gen(&__MODULE__.and_gate/2)
    or_training_examples = example_gen(&__MODULE__.or_gate/2)
    nor_training_examples = example_gen(&__MODULE__.nor_gate/2)
    nand_training_examples = example_gen(&__MODULE__.nand_gate/2)

    [
      {and_training_examples, "AND"},
      {or_training_examples, "OR"},
      {nand_training_examples, "NAND"},
      {nor_training_examples, "NOR"}
    ]
    |> Enum.each(fn {training_examples, gate} ->
      IO.puts("")
      IO.puts("==== FOR #{gate} gate ====")
      start(training_examples)
    end)
  end

  def start(training_examples) do
    # Initial theta value
    theta = theta_init(3)

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
    |> Enum.reduce(weights, fn {inp1, inp2, output}, [w1, w2, w3] ->
      h = sigmoid(w1 * 1 + w2 * inp1 + w3 * inp2)

      error = output - h

      d_output = error * sigmoid_grad(h)

      w1 = w1 + 1 * d_output * @alpha
      w2 = w2 + inp1 * d_output * @alpha
      w3 = w3 + inp2 * d_output * @alpha

      [w1, w2, w3]
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
      [w1, w2, w3] = weights
      prediction = sigmoid(w1 * 1 + w2 * inp1 + w3 * inp2)

      output = if prediction < 0.5, do: 0, else: 1

      IO.puts("For input #{inp1}, #{inp2} , PREDICTION = #{output}")
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
