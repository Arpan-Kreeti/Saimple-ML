# Simple, single feature Linear Regression implementation

defmodule LinearRegression do
  def start do
    # Get training data
    {x1, y1} = Data.training_data()
    # Get test data
    {x2, y2} = Data.test_data()
    # Trainthe model/ find the values of theta
    {theta0, theta1} = GradientDescent.descent(x1, y1, 0, 0, 0)
    #{theta0, theta1} = {2.9266145660124283, 0.510701696042587}
    # Make prediction on test data set ( 2.9266145660124283, 0.5107016960425879)
    LinearRegression.predict(x2, y2, theta0, theta1)
  end

  def hypothesis(theta0, theta1, x) do
    theta0 + theta1 * x
  end

  def gradient(theta0, theta1, x, y, m, j) do
    tmp = 1 / m

    sum =
      Enum.zip(x, y)
      |> Enum.reduce(0, fn {x1, y1}, acc ->
        error =
          if(j == 0,
            do: LinearRegression.hypothesis(theta0, theta1, x1) - y1,
            else: (LinearRegression.hypothesis(theta0, theta1, x1) - y1) * x1
          )

        acc + error
      end)

    tmp * sum
  end

  def cost(x, y, m, theta0, theta1) do
    tmp = 1 / (2 * m)

    sum =
      Enum.zip(x, y)
      |> Enum.reduce(0, fn {x1, y1}, acc ->
        error = hypothesis(theta0, theta1, x1) - y1

        acc + :math.pow(error, 2)
      end)

    tmp * sum
  end

  def predict(x, y, theta0, theta1) do
    Enum.zip(x, y)
    |> Enum.each(fn {x, y} ->
      prediction = hypothesis(theta0, theta1, x)
      error = abs((y - prediction) *100 / y) |> Float.round(2)

      IO.puts("For x = #{x}, y = #{y}, prediction = #{prediction}, error = #{error}%")
    end)
  end
end

defmodule GradientDescent do
  # Learning Rate
  @alpha 0.01
  # Criteria for stopping Gradient descent, target value for cost function
  @epsilon 0.1
  # Max gradient descent iteration
  @max_iter 5000
  # Number of training examples
  @m 30

  def descent(x, y, theta0, theta1, iter) do
    if(iter < @max_iter) do
      IO.puts("Interation: #{iter}")
      IO.puts("Value of theta0, theta1 = #{theta0}, #{theta1}")
      cost = LinearRegression.cost(x, y, @m, theta0, theta1)
      IO.puts("Value of cost function = #{cost}")

      grad0 = LinearRegression.gradient(theta0, theta1, x, y, @m, 0)
      grad1 = LinearRegression.gradient(theta0, theta1, x, y, @m, 1)

      IO.puts("Grad0 = #{grad0}, Grad1 = #{grad1}")
      IO.puts("")

      if(cost < @epsilon) do
        {theta0, theta1}
      else
        tmp0 = theta0 - @alpha * grad0
        tmp1 = theta1 - @alpha * grad1

        descent(x, y, tmp0, tmp1, iter + 1)
      end
    else
      {theta0, theta1}
    end
  end
end

defmodule Data do
  def training_data do
    {Enum.take(dataX(), 30), Enum.take(dataY(), 30)}
  end

  def test_data do
    {Enum.slice(dataX(), 30, 44), Enum.slice(dataY(), 30, 44)}
  end

  def dataX do
    data()
    |> Enum.with_index()
    |> Enum.filter(fn {_, index} -> rem(index, 2) == 0 end)
    |> Enum.map(fn {e, _} -> e end)
  end

  def dataY do
    data()
    |> Enum.with_index()
    |> Enum.filter(fn {_, index} -> rem(index, 2) != 0 end)
    |> Enum.map(fn {e, _} -> e end)
  end

  def data do
    [
      10.0,
      8.04,
      8.0,
      6.95,
      13.0,
      7.58,
      9.0,
      8.81,
      11.0,
      8.33,
      14.0,
      9.96,
      6.0,
      7.24,
      4.0,
      4.26,
      12.0,
      10.84,
      7.0,
      4.82,
      5.0,
      5.68,
      10.0,
      9.14,
      8.0,
      8.14,
      13.0,
      8.74,
      9.0,
      8.77,
      11.0,
      9.26,
      14.0,
      8.1,
      6.0,
      6.13,
      4.0,
      3.1,
      12.0,
      9.13,
      7.0,
      7.26,
      5.0,
      4.74,
      10.0,
      7.46,
      8.0,
      6.77,
      13.0,
      12.74,
      9.0,
      7.11,
      11.0,
      7.81,
      14.0,
      8.84,
      6.0,
      6.08,
      4.0,
      5.39,
      12.0,
      8.15,
      7.0,
      6.42,
      5.0,
      5.73,
      8.0,
      6.58,
      8.0,
      5.76,
      8.0,
      7.71,
      8.0,
      8.84,
      8.0,
      8.47,
      8.0,
      7.04,
      8.0,
      5.25,
      19.0,
      12.5,
      8.0,
      5.56,
      8.0,
      7.91,
      8.0,
      6.89
    ]
  end
end

# defmodule Data do
#   def training_data do
#     {Enum.take(dataX(), 70), Enum.take(dataY(), 70)}
#   end

#   def test_data do
#     {Enum.slice(dataX(), 70, 97), Enum.slice(dataY(), 70, 97)}
#   end

#   def dataX do
#     [
#       6.1101,
#       5.5277,
#       8.5186,
#       7.0032,
#       5.8598,
#       8.3829,
#       7.4764,
#       8.5781,
#       6.4862,
#       5.0546,
#       5.7107,
#       14.164,
#       5.734,
#       8.4084,
#       5.6407,
#       5.3794,
#       6.3654,
#       5.1301,
#       6.4296,
#       7.0708,
#       6.1891,
#       20.27,
#       5.4901,
#       6.3261,
#       5.5649,
#       18.945,
#       12.828,
#       10.957,
#       13.176,
#       22.203,
#       5.2524,
#       6.5894,
#       9.2482,
#       5.8918,
#       8.2111,
#       7.9334,
#       8.0959,
#       5.6063,
#       12.836,
#       6.3534,
#       5.4069,
#       6.8825,
#       11.708,
#       5.7737,
#       7.8247,
#       7.0931,
#       5.0702,
#       5.8014,
#       11.7,
#       5.5416,
#       7.5402,
#       5.3077,
#       7.4239,
#       7.6031,
#       6.3328,
#       6.3589,
#       6.2742,
#       5.6397,
#       9.3102,
#       9.4536,
#       8.8254,
#       5.1793,
#       21.279,
#       14.908,
#       18.959,
#       7.2182,
#       8.2951,
#       10.236,
#       5.4994,
#       20.341,
#       10.136,
#       7.3345,
#       6.0062,
#       7.2259,
#       5.0269,
#       6.5479,
#       7.5386,
#       5.0365,
#       10.274,
#       5.1077,
#       5.7292,
#       5.1884,
#       6.3557,
#       9.7687,
#       6.5159,
#       8.5172,
#       9.1802,
#       6.002,
#       5.5204,
#       5.0594,
#       5.7077,
#       7.6366,
#       5.8707,
#       5.3054,
#       8.2934,
#       13.394,
#       5.4369
#     ]
#   end

#   def dataY do
#     [
#       17.592,
#       9.1302,
#       13.662,
#       11.854,
#       6.8233,
#       11.886,
#       4.3483,
#       12,
#       6.5987,
#       3.8166,
#       3.2522,
#       15.505,
#       3.1551,
#       7.2258,
#       0.71618,
#       3.5129,
#       5.3048,
#       0.56077,
#       3.6518,
#       5.3893,
#       3.1386,
#       21.767,
#       4.263,
#       5.1875,
#       3.0825,
#       22.638,
#       13.501,
#       7.0467,
#       14.692,
#       24.147,
#       -1.22,
#       5.9966,
#       12.134,
#       1.8495,
#       6.5426,
#       4.5623,
#       4.1164,
#       3.3928,
#       10.117,
#       5.4974,
#       0.55657,
#       3.9115,
#       5.3854,
#       2.4406,
#       6.7318,
#       1.0463,
#       5.1337,
#       1.844,
#       8.0043,
#       1.0179,
#       6.7504,
#       1.8396,
#       4.2885,
#       4.9981,
#       1.4233,
#       -1.4211,
#       2.4756,
#       4.6042,
#       3.9624,
#       5.4141,
#       5.1694,
#       -0.74279,
#       17.929,
#       12.054,
#       17.054,
#       4.8852,
#       5.7442,
#       7.7754,
#       1.0173,
#       20.992,
#       6.6799,
#       4.0259,
#       1.2784,
#       3.3411,
#       -2.6807,
#       0.29678,
#       3.8845,
#       5.7014,
#       6.7526,
#       2.0576,
#       0.47953,
#       0.20421,
#       0.67861,
#       7.5435,
#       5.3436,
#       4.2415,
#       6.7981,
#       0.92695,
#       0.152,
#       2.8214,
#       1.8451,
#       4.2959,
#       7.2029,
#       1.9869,
#       0.14454,
#       9.0551,
#       0.61705
#     ]
#   end
# end
