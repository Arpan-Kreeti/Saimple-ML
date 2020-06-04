defmodule NaiveBayes do
  def predict(file_path, predictors) do
  end

  def load_csv(prediction_var \\ []) do
    file_path = "data/weather.csv"

    [headers | data] =
      file_path
      |> File.stream!()
      |> CSV.decode!()
      |> Enum.to_list()

    data_count = Enum.count(data)

    classes =
      data
      |> Enum.reduce([], fn [class | _rest], acc ->
        acc ++ [class]
      end)
      |> Enum.uniq()

    class_map = classes |> Enum.reduce(%{}, fn class, acc -> Map.put(acc, class, 0) end)

    class_map =
      Enum.reduce(data, class_map, fn [class | _rest], class_map ->
        count = Map.get(class_map, class) + 1

        Map.put(class_map, class, count)
      end)
      |> Enum.reduce(%{}, fn {key, val}, acc ->
        Map.put(acc, key, val / data_count)
      end)

    [class_header | predictors] = headers

    counts =
      headers
      |> Enum.with_index()
      |> Enum.reduce(%{}, fn {header, index}, acc ->
        count =
          data
          |> Enum.count(fn row ->
            Enum.at(row, index) == "1"
          end)

        Map.put(acc, header, count)
      end)
      |> Map.delete(class_header)

    cond_probs =
      Enum.reduce(class_map, %{}, fn {class, prob}, acc ->
        cond_prod =
          predictors
          |> Enum.with_index()
          |> Enum.map(fn {predictor, index} ->
            count =
              Enum.count(data, fn [curr_class | rest] ->
                class == curr_class && Enum.at(rest, index) == "1"
              end)

            curr_prob = count / data_count

            {predictor, curr_prob / prob}
          end)
          |> Enum.reduce(%{}, fn {predictor, count}, acc ->
            Map.put(acc, predictor, count)
          end)

        Map.put(acc, class, cond_prod)
      end)

      # cond_probs has conditional probabilitys
      require IEx
      IEx.pry

    prepared_prediction_vars =
      Enum.zip(predictors, prediction_var)
      |> Enum.filter(fn {_var, bit} -> bit == "1" end)
      |> Enum.map(fn {var, _bit} -> var end)

    true_predictors_count =
      data
      |> Enum.count(fn [_class | rest] ->
        prediction_var == rest
      end)

    true_predictors_prob = true_predictors_count / data_count

    output =
      Enum.map(classes, fn class ->
        temp =
          prepared_prediction_vars
          |> Enum.reduce(1, fn predictor, product ->
            product * cond_probs[class][predictor]
          end)

        class_prob = temp * class_map[class] / true_predictors_prob

        {class, class_prob}
      end)

    IO.puts("Classes Identified:")
    IO.inspect(class_map)

    IO.puts("--------------------")

    IO.puts("Conditional Probabilitys:")
    IO.inspect(cond_probs)

    IO.puts("--------------------")

    IO.puts("Number of data: #{data_count}")

    IO.inspect(output)
  end
end
