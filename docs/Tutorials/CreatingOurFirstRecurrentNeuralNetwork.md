# Creating Our First Recurrent Neural Network

In this tutorial, we will show you on how to create the recurrent neural network variant and give it the ability to handle temporal sequencing.

Before we proceed this tutorial, you are required to read these two tutorials:

* [Creating Our First Neural Network](CreatingOurFirstNeuralNetwork.md)

* [General Tensor Conventions](GeneralTensorConventions.md)

# Setting Up The Recurrent Neural Network Cell

In the previous tutorial, you have seen that the we have to manually create our weight tensors and WeightContainer. Fortunately, because recurrent neural networks and its variants have specific configurations, this library provides you a modular way to create it.

```lua

local RecurrentModels = DataPredictAxon.RecurrentModels

local RNNCell, WeightContainer, reset, setHiddenStateTensor = RecurrentModels.RecurrentNeuralNetworkCell{inputSize = 1, hiddenSize = 1, learningRate = 0.001, activationFunction = "FastLeakyRectifiedLinearUnit"}

-- For this tutorial, we will change the activationFunction to FastLeakyRectifiedLinearUnit instead of using the default.
-- You can use hidden size to determine the maximum number of features it should output.

```

Now, notice that this is recurrent neural network "cell", which means that it can only handle a single timestep. To change this, we must "uncell" our recurrent neural network.

```lua

local RNN = RecurrentModels.UncellModel{RNNCell, true} 

-- Setting the second parameter to true will make it train in reverse sequence.

```

Once you have everything set up, you can test this recurrent neural network with some data.

```lua

local CostFunctions = DataPredictAxon.CostFunctions

local inputTensor = {

	{{1}, {2}, {3}}, 
	{{2}, {3}, {4}}, 
	{{3}, {4}, {5}}, 
	{{4}, {5}, {6}}, 
	{{5}, {7}, {8}}

}

local outputTensor = {

	{{2}, {3}, {4}},
	{{3}, {4}, {5}}, 
	{{4}, {5}, {6}}, 
	{{5}, {6}, {7}}, 
	{{6}, {8}, {9}}

}

for i = 1, 300, 1 do
	
	local generatedLabelTensor = RNN{inputTensor}
	
	local costTensor = CostFunctions.FastMeanSquaredError{generatedLabelTensor, outputTensor}
	
	costTensor:differentiate()

	costTensor:destroy()
	
	reset() -- We need to reset the hidden stete tensor before going to the next iteration.
	
	task.wait()
	
	WeightContainer:gradientDescent()
	
end

for i = 10, 100, 1 do
	
	local test3 = RNNCell{{{i}}} -- Notice that this is a "cell", hence you must only give a single timestep.
	
	print(test3)
	
end

```

Whew! That took quite a bit. 

You may have noticed why we don't manually handle the hidden state tensors like other deep learning libraries and frameworks. This is because DataPredict™ Axon is designed to reduce the manual handling that could be otherwise be a headache to keep track on. As such, this allows any recurrent neural networks under DataPredict™ Axon to be extended easily.

That's all for today and have fun with it!
