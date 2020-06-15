defmodule NeuralNet do
  @theta1 [[-30, 20, 20], [10, -20, -20]]
  @theta2 [[-10, 20, 20]]

  def xnor_forward_prop(x1, x2) do
    # Input, -1 is the bias
    a0 = [1, x1, x2]

    [a11, a12] = calc_new_layer(a0, @theta1, 2)

    IO.puts("Output of Layer 1: #{inspect([a11, a12])}")

    # Add bias to 2nd layer
    a2 = [1, a11, a12]

    [a3] = calc_new_layer(a2, @theta2, 1)

    IO.puts("Output of XNOR on input (#{x1}, #{x2}) = #{transform(a3)}")
  end

  def transform(x) when x < 0.5, do: 0

  def transform(_), do: 1

  def calc_new_layer(x, theta, number_of_units) do
    1..number_of_units
    |> Enum.zip(theta)
    |> Enum.map(fn {_i, theta_i} ->
      # Calculate ith activation
      calc_activation(x, theta_i)
    end)

  end

  def calc_activation(x, theta) do
    if Enum.count(theta) != Enum.count(x) do
      raise "Incorrect dimensions !"
    end

    Enum.zip(theta, x)
    |> Enum.reduce(0, fn {a, b}, acc ->
    #IO.puts "#{a} * #{b}"
      acc + a * b
    end)
    |> sigmoid()
  end

  def sigmoid(x) do
    1 / (1 + :math.exp(-x))
  end
end
