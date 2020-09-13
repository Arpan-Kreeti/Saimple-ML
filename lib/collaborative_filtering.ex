defmodule CollaborativeFiltering do
  import Matrix

  def input do
    # 4 users: Random user preferences (action, romantic)
    random_theta = [
      [0.031641796892049934, 1 - 0.031641796892049934],
      [0.41337976300567036, 1 - 0.41337976300567036],
      [0.2721034180199562, 1 - 0.2721034180199562],
      [0.7650338835924677, 1 - 0.7650338835924677]
    ]

    # random_theta = [
    #   [0, 5]
    # ]

    # random_inputs = [
    #   [0, 5],
    #   [3, 1]
    # ]

    # 5 movies: random movie features (action, romantic)
    random_inputs = [
      [0.23878965334399904, 0.9744610278706924],
      [0.4276564332056534, 0.33807536831160023],
      [0.8113699277397208, 0.30607450060968766],
      [0.2533249447088597, 0.4002474119791435],
      [0.5577928481165189, 0.3002371551465468]
    ]

    # theta = [
    #   [5, 0],
    #   [5, 0],
    #   [0, 5],
    #   [0, 5]
    # ]

    y = [
      [5, 5, 0, 0],
      [5, :nan, :nan, 0],
      [:nan, 4, 0, :nan],
      [0, 0, 5, 4],
      [0, 0, 5, :nan]
    ]

    cost_function(random_theta, random_inputs, y)
  end

  def cost_function(theta, inputs, y) do
    # for each user
    theta
    |> Enum.map(fn [w1, w2] ->
      # for each movie
      inputs
      |> Enum.map(fn [x1, x2] ->
        w1 * x1 + w2 * x2
      end)
    end)
    |> Enum.zip(y)
    |> Enum.map(fn {predicted, actual} ->
      Enum.zip(predicted, actual)
      |> Enum.map(fn {predicted, actual} ->
        if actual == :nan, do: 0, else: actual - predicted
      end)
    end)
    |> Enum.each(&IO.inspect/1)
  end
end
