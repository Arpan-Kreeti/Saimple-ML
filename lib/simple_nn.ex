defmodule NeuralNet do
  #@theta1 [[-30, 20, 20], [10, -20, -20]]
  #@theta2 [[-10, 20, 20]]

  @theta1 [[1.1800000000000005e-4, 9.659999999999998e-4, -4.4799999999999994e-4], [-1.4400000000000003e-4, -1.8600000000000008e-4, 5.18e-4]]
  @theta2 [[-0.006750396491856032, -0.006582396491856032, -0.005994396491856034]]

  def xnor_forward_prop(x1, x2) do
    # Input, 1 is the bias
    a1 = [1, x1, x2]

    [a11, a12] = calc_new_layer(a1, @theta1, 2)

    IO.puts("Output of Layer 1: #{inspect([a11, a12])}")

    # Add bias to 2nd layer@INIT_EPSILON
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
      # IO.puts "#{a} * #{b}"
      acc + a * b
    end)
    |> sigmoid()
  end

  def sigmoid(x) do
    1 / (1 + :math.exp(-x))
  end
end

defmodule Backprop do
  @init_epsilon 1.0e-3
  alias NeuralNet, as: Nn

  def start(m, training_set, theta1, theta2) do
    # theta1 = [[-30, 20, 20], [10, -20, -20]]
    # theta2 = [[-10, 20, 20]]

    # Perform Forward Prop, and calculate activation unit values for each input in each layer
    activations =
      Enum.reduce(training_set, [], fn {inp_1, inp_2, output}, acc ->
        # Add bias to first layer or input layer
        a1 = [1, inp_1, inp_2]

        [a11, a12] = Nn.calc_new_layer(a1, theta1, 2)

        # Add bias to 2nd layer
        a2 = [1, a11, a12]

        a3 = Nn.calc_new_layer(a2, theta2, 1)

        # IO.puts("Actual Output: #{output}, Predcited Value: #{[t]=a3; if(t < 0.5001530169294726, do: 0, else: 1)}")

        acc ++ [{a1, a2, a3, output}]
      end)

    cost = calc_cost(m, activations)

    {theta1_grad, theta2_grad} =
      Enum.reduce(training_set, {0, 0}, fn {inp_1, inp_2, output}, {theta1_grad, theta2_grad} ->
        a1 = [1, inp_1, inp_2]
        [a11, a12] = Nn.calc_new_layer(a1, theta1, 2)

        a2 = [1, a11, a12]

        [a3] = Nn.calc_new_layer(a2, theta2, 1)

        delta3 = a3 - output

        z2 =
          1..2
          |> Enum.zip(theta1)
          |> Enum.map(fn {_i, theta_i} ->
            Enum.zip(theta_i, a1)
            |> Enum.reduce(0, fn {a, b}, acc ->
              acc + a * b
            end)
          end)

        z2 = [1] ++ z2

        #  delta2 = (Theta2' * delta3) .* sigmoidGradient(z2)

        sigmoid_grad =
          Enum.map(z2, fn x ->
            x * (1 - x)
          end)

        temp = Enum.map(z2, fn x -> x * delta3 end)

        delta2 = Enum.zip(temp, sigmoid_grad) |> Enum.map(fn {a, b} -> a * b end)

        # Ignore error in first layer
        [_ | delta2] = delta2

        temp_a2 = Enum.slice(a2, 1, 2)

        theta1_grad =
          theta1_grad +
            (Enum.zip(delta2, temp_a2) |> Enum.reduce(0, fn {a, b}, acc -> acc + a * b end))

        theta2_grad =
          theta2_grad + Enum.reduce(Enum.slice(a1, 1, 2), 0, fn x, acc -> acc + x * delta3 end)

        {theta1_grad, theta2_grad}
      end)

    theta1_grad = 1 / m * theta1_grad
    theta2_grad = 1 / m * theta2_grad

    {cost, theta1_grad, theta2_grad}
  end

  def example_gen(func) do
    1..1000
    |> Enum.map(fn _ ->
      rand_1 = rand()
      rand_2 = rand()

      output = func.(rand_1, rand_2)

      {rand_1, rand_2, output}
    end)
  end

  def rand do
    :rand.uniform(100_000)
    |> rem(2)
  end

  def xnor(inp_1, inp_2) do
    if(inp_1 == inp_2, do: 1, else: 0)
  end

  # Gives us a random value between [-init_epsilon, +init_epsilon]
  def theta_init(n) do
    Enum.reduce(1..n, [], fn _, acc ->
      rand = :rand.uniform(999) / 1000 * (2 * @init_epsilon) - @init_epsilon
      acc ++ [rand]
    end)
  end

  def calc_cost(m, list) do
    # Cost function: -1/m * ∑(1 to m) [y*log(h) + (1-y)log(1-h)]
    temp =
      Enum.reduce(list, 0, fn {_a1, _a2, [a3], output}, acc ->
        acc + (output * :math.log(a3) + (1 - output) * :math.log(1 - a3))
      end)

    -1 / m * temp
  end
end

defmodule NGradientDescent do
  # Learning Rate
  @alpha 0.001
  # Criteria for stopping Gradient descent, target value for cost function
  @epsilon 0.1
  # Max gradient descent iteration
  @max_iter 5000

  def start do
    m = 1000
    training_set = Backprop.example_gen(&Backprop.xnor/2)

    theta1 = [Backprop.theta_init(3), Backprop.theta_init(3)]

    theta2 = [Backprop.theta_init(3)]

    # (m, training_set, theta1, theta2)

    descent(&Backprop.start/4, training_set, m, theta1, theta2, 0)
  end

  def descent(cost_func, training_set, m, theta1, theta2, iter) do
    if(iter < @max_iter) do
      # IO.puts("Interation: #{iter}")
      # IO.puts("Value of theta0, theta1 = #{theta1}, #{theta2}")
      {cost, grad1, grad2} = cost_func.(m, training_set, theta1, theta2)

      if(rem(iter, 100) == 0) do
        IO.puts("-- ITERATION: #{iter} --")
        IO.puts("Value of cost function = #{cost}")
        IO.puts("Grad1 = #{grad1}, Grad2 = #{grad2}")
        IO.puts("Theta1: #{inspect(theta1)}")
        IO.puts("")
        IO.puts("Theta2: #{inspect(theta2)}")
        IO.puts("")
      end

      if(cost < @epsilon) do
        {theta1, theta2}
      else
        tmp0 = Enum.map(theta1, fn x -> Enum.map(x, fn y -> y - @alpha * grad1 end) end)
        tmp1 = Enum.map(theta2, fn x -> Enum.map(x, fn y -> y - @alpha * grad2 end) end)

        descent(cost_func, training_set, m, tmp0, tmp1, iter + 1)
      end
    else
      {theta1, theta2}
    end
  end
end
