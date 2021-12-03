# Ben Housley / Lincoln Holt
# bhd1509 / ljh2306
from neural import NeuralNet
import pandas
import csv
# each row is an (input, output) tuple

xor_data = [ 
    #   input     output    corresponding example 
    ([0.0, 0.0],  [0.0]),  #[0, 0] => 0 
    ([0.0, 1.0],  [1.0]),  #[0, 1] => 1 
    ([1.0, 0.0],  [1.0]),  #[1, 0] => 1 
    ([1.0, 1.0],  [0.0])   #[1, 1] => 0 
]

# Input, Hidden, Output
# FIRST RUN
nn = NeuralNet(2, 5, 1) 
nn.train(xor_data)
# nn.evaluate
# nn.test

# SECOND RUN
nn = NeuralNet(2, 1, 1) 
nn.train(xor_data)
print(nn.test_with_expected(xor_data))

# THIRD RUN
wine_data = pandas.read_csv("wine.data")

normalized_wine = (wine_data - wine_data.min()) / (wine_data.max()-wine_data.min())
wine_list = normalized_wine.values.tolist()

# [(input list, output list), (input_list, output_list), ... ]
wine_formatted = []
for line in wine_list:
    output = line[0]
    features = line[1:]
    wine_formatted.append((features, [output]))
print(wine_formatted)
wine_net = NeuralNet(13, 15, 1)
wine_net.train(wine_formatted, momentum_factor=.2, learning_rate=.6, iters=1000)
result = wine_net.test_with_expected(wine_formatted)

for line in result:
    print('Desired: {}, Actual: {}'.format(line[1], line[2]))




