# Creating Our First Recurrent Neural Network

From our previous [tutorial](CreatingOurFirstNeuralNetwork.md), we have seen how simple it is to build a regular neural network. I recommend you to read the previous tutorial if you have not done so already just to make the current tutorial easier to understand. 

In this tutorial, we will show on how to create the recurrent neural network variant and give it the ability to handle temporal sequencing.

# Setting Up The Recurrent Neural Network Cell

In the previous tutorial, you have seen that the we have to manually create our weight tensors and WeightContainer. Fortunately, because recurrent neural networks and its variants have specific configurations, this library provides you a modular way to setup it.

```lua

local RecurrentModels = DataPredictAxon.RecurrentModels

local RNNCell, WeightContainer, reset, setHiddenStateTensor = RecurrentModels.RecurrentNeuralNetworkCell{inputSize = 1, hiddenSize = 1, learningRate = 0.001, activationFunction = "FastLeakyRectifiedLinearUnit"}

-- Generally, this model has FastSigmoid as a default for activationFunction parameter. For this tutorial, we will change to FastLeakyRectifiedLinearUnit activation function.
-- You can use hidden size to determine the maximum number of features it should output.

```

Now, notice that this is recurrent neural network "cell", which means that it can only handle a single timestep. To change this, we must "uncell" our recurrent neural network.

```lua

local RNN = RecurrentModels.UncellModel{RNNCell, true} 

-- Setting the second parameter to true will make it train in reverse sequence.

```
