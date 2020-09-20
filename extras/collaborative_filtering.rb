# COLLABORATIVE FILTERING FOR MOVIE RATING PREDICTION

require 'matrix'

MAX_ITER = 5000
EPSILON = 0.01
ALPHA = 0.01

def training_data

    movie = {
            "Love at last" => [5, 3],
            "Romance forever" => [5, 0],
            "Cute puppies of love" => [5, 1], 
            "Sword vs. karate" => [0, 5],
            "Nonstop car chases" => [1, 5]
        }

    people = {
                "Alice" => [5, 0],
                "Bob" => [5, 3],
                "Carol" => [0, 4],
                "Dave" => [0, 1]
            }

            # 4 * 5
    ratings = Matrix[
        [25, 25, 25, 5, 0], 
        [34, 25, 28, 20, 15], 
        [12, 0, 4, 20, 20], 
        [3, 0, 1, 5, 5]
    ]

    return [movie, people, ratings]

end


def random_initial
    # 4 * 2
    random_theta = Matrix[
        [0.031641796892049934, 1 - 0.031641796892049934],
        [0.41337976300567036, 1 - 0.41337976300567036],
        [0.2721034180199562, 1 - 0.2721034180199562],
        [0.7650338835924677, 1 - 0.7650338835924677]
        ]

    # 5 * 2
    random_x = Matrix[
        [0.23878965334399904, 0.9744610278706924],
        [0.4276564332056534, 0.33807536831160023],
        [0.8113699277397208, 0.30607450060968766],
        [0.2533249447088597, 0.4002474119791435],
        [0.5577928481165189, 0.3002371551465468]
        ]

    [random_theta, random_x]
end



def cost(theta, x, y)
    hypothesis = (theta * x.transpose)

    error = hypothesis - y

    square_error = 0.5 * error.collect { |x| x * x }

    square_error.sum
end

def descent(x, theta, y, iter = 1)

    if (iter > MAX_ITER) 
        return [theta, x]
    end

    puts("Iteration: #{iter}")

    cost = cost(theta, x, y)
    puts("COST = #{cost}")


    if (cost < EPSILON) 
        return [theta, x]
    end

    delta_x, delta_theta = gradient(x, theta, y)

    x = x - ALPHA * delta_x
    theta =  theta - ALPHA * delta_theta

    descent(x, theta, y, iter + 1)
end

def gradient(x, theta, y)
    delta_x = ((theta * x.transpose) - y).transpose * theta
    delta_theta = ((theta * x.transpose) - y) * x
    return [delta_x, delta_theta]
end

def start 

    movies, people, ratings = training_data
    random_theta, random_x = random_initial

    #Train and learn theta and X
    learned_theta, learned_x = descent(random_x, random_theta, ratings)

    puts""
    puts"============= TRAINING DONE ==============="
    puts ""
    puts "Enter a new Movie"
    print "Movie Name: "
    name = gets.chomp
    print "Romantic Rating(0 to 5): "
    x1 = gets.chomp.to_f
    print "Action Rating(0 to 5): "
    x2 = gets.chomp.to_f

    prediction = learned_theta * Matrix[[x2, x1]].transpose

    people.keys.zip(prediction).each { |e| puts "#{e[0]} would rate #{e[1]}" }

    puts"-------------------------------------------"
    puts ""

    puts "Enter a new person"
    print "Person Name: "
    name = gets.chomp
    print "Romantic Likings(0 to 5): "
    x1 = gets.chomp.to_f
    print "Action Likings(0 to 5): "
    x2 = gets.chomp.to_f

    prediction = learned_x * Matrix[[x2, x1]].transpose

    movies.keys.zip(prediction).each { |e| puts "#{e[0]} would be rated #{e[1]}" }
end

start()
