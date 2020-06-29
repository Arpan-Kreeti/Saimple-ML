# THETA_1 = [[-30, 20, 20], [10, -20, -20]]
# THETA_2 = [[-10, 20, 20]]

THETA_1 = [[0.033364540155802654, 0.008974423924929458, -0.028347557481134426], [-0.05933696505075069, 0.10401574814616084, -0.03433588323867402]]

THETA_2 =  [[-0.062727321743182, 0.003281470102241176, -0.08076178866148431]]



def xnor_forward_prop(input1, input2)

  # Add bias to inputs
  a1 = [1, input1, input2]

  a11, a12 = calc_activation(a1, THETA_1)

  # Add bias to 2nd layer activation units
  a2 = [1, a11, a12]

  a3 = calc_activation(a2, THETA_2)

  a3 = a3.first

  output = transform(a3)

  puts("Output of XNOR on input (#{input1}, #{input2}) = #{output}")
end

def calc_activation(input, theta)
  theta.map do |curr_theta|
    tmp =curr_theta.zip(input).reduce(0) do |acc, nums|
      x,y = nums
      acc + (x * y)
    end
    sigmoid(tmp)
  end
end

def transform(prediction)
  prediction < 0.5 ? 0 : 1
end

def sigmoid(value)
  1.0 / (1.0 + Math.exp(-value))
end

xnor_forward_prop(0,0)
xnor_forward_prop(0,1)
xnor_forward_prop(1,0)
xnor_forward_prop(1,1)