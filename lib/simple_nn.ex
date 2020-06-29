defmodule SimpleNeuralNet do
  @iter 5000
  @init_epsilon 0.12
  @m 1000

  def start do
    training_examples = example_gen(&SimpleNeuralNet.and_gate/2)

    theta = theta_init(3)

    theta =
      1..@iter
      |> Enum.reduce(theta, fn _, theta ->
        [w1, w2, w3] = theta

        [d_bias, d_inp1, d_inp2] =
          training_examples
          |> Enum.reduce([0, 0, 0], fn {inp1, inp2, output}, theta ->
            h = sigmoid(w1 * 1 + w2 * inp1 + w3 * inp2)
            error = h - output

            [dw1, dw2, dw3] = theta

            delta_bias = error * sigmoid_grad(1)
            delta_1 = error * sigmoid_grad(inp1)
            delta_2 = error * sigmoid_grad(inp2)

            d_bias = dw1 + 1 * delta_bias
            d_inp1 = dw2 + inp1 * delta_1
            d_inp2 = dw3 + inp2 * delta_2

            [d_bias, d_inp1, d_inp2]
          end)

        tmp = 1 / @m

        [w1 + tmp * d_bias, w2 + tmp * d_inp1, w3 + tmp * d_inp2]
      end)

    IO.inspect(theta)
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

  def and_gate(inp_1, inp_2) do
    if inp_1 == 1 && inp_2 == 1, do: 1, else: 0
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
