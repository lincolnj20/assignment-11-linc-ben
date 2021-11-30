# Ben Housley / Lincoln Holt
# bhd1509 / ljh2306
from neural import NeuralNet
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
