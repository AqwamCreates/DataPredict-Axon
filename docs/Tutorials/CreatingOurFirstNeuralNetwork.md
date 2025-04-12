# Creating Our First Neural Network

In this tutorial, we will show you on how to create the most basic neural network. We will split this tutorial into multiple smaller sections so that you do not get overwhelmed with too much information.

Here are the list of section for you to warm up before reading through them.

* Gentle introduction automatic differentiation tensors
	
* Setting up the weights
	
* Choosing a cost function
	
* Building the model
	
## Gentle Introduction To Automatic Differentiation Tensors

Automatic differentiation tensors (or ADTensors) are the building blocks for neural networks. These ADTensors handles the following tasks:

* Transforming inputs to certain outputs.
	
* Calculating first-order derivatives for a given value.
	
* Storing the inputs, transformed inputs and first-order derivative values.
	
ADTensors are typically used by:

* Activation Functions: Converts values into another.
	
* Weights Tensors: Holds the weights for our neural network.

* And many others!

## Setting Up The Weights

In order for our neural network to learn, we need to store the weight values for our neural network. These are typically stored in the weight ADTensors. In order to create them, we first need to set the dimension sizes of the weights. 

Below, it shows a code creating a layer block containing a weight tensor of specific size. It is also setting up learning rate value when the neural network performs a gradient descent.

```lua

local ADTensor = DataPredictNeural.AutomaticDifferentiationTensor

local WeightADTensor = ADTensor.new{dimensionSizeArray = {1, 90, 4}} -- Pay attention to the fact we're using curly brackets and not the normal brackets when inputting out parameters.

-- Alternatively, we can use the function parameters position to implicitly tell what type of value for that particular value.

local WeightADTensor = ADTensor.new{{1, 90, 4}}

```

As you can see, the length of dimension array is equal to total number of dimensions, and the values inside it represent the sizes of each dimension.

## Choosing A Cost Function

The cost function tells us the performance of the model making predictions compared to the label values. When the predictions are close to the label values, the cost is low. Otherwise, the cost is high. By minimizing this cost, the network gets better at capturing the patterns in our data and making more accurate predictions.

Below, we will show a commented code related to cost function.

```lua

local CostFunction = DataPredictNeural.CostFunctions
	
local costValue = CostFunction.MeanSquaredError{generatedLabelTensor, labelTensor} -- This function is used to calculate the overall cost or error between the output and label tensors.

```

With the fundamental knowledge in place, we can now create our first neural network.

## Building The Model

Below, we will show you a block of code and describe what each line of code are doing through the comments.

```lua

local ServerScriptService = game.ServerScriptService

local TensorL = require(ServerScriptService.TensorL_Table_Nested_Efficient_Version_2)

local DataPredictAxon = require(ServerScriptService.DataPredictAxon)

local ADTensor = DataPredictAxon.AutomaticDifferentiationTensor

local PaddingLayer = DataPredictAxon.PaddingLayers

local inputTensor = ADTensor.createRandomNormalTensor{{10, 2}}

local weightTensor = ADTensor.createRandomNormalTensor{{2, 4}}

local targetTensor = ADTensor.createRandomNormalTensor{{10, 4}}

local WeightContainer = DataPredictAxon.WeightContainer.new{ -- This allows us to adjust the weights.

	{weightTensor, 1},

}

--[[

In order for us to be able to calculate the loss tensor, we need to make sure the generated label tensor dimensions matches with the original one.

When initializing the weights, ensure that the 2nd dimension of the input tensor matches the 1st dimension of the weight tensor.

When doing the dot product between the input tensor and weight tensor, it will give a new tensor shape.

	* Input tensor: {a, b}
	
	* Weight tensor: {b, c}
	
	* Output tensor: {a, c}

--]]

local function model(inputTensorPlaceHolder, weightTensorPlaceHolder) -- Let's create ourselves a good old model in a form of function.
	
	local tensor3 = inputTensorPlaceHolder:dotProduct{WeightTensorPlaceHolder}

	local finalTensor = DataPredictAxon.ActivationFunctionLayers.Sigmoid{tensor4}

	local costValue = DataPredictAxon.CostFunctions.MeanSquaredError{finalTensor, targetTensor}
	
	return costValue
	
end

for i = 1, 100000 do

	local costValue = model(tensor1, tensor2)
	
	costValue:differentiate() -- Calling this will calculate the first derivative tensors for all our operations, including for out weight tensor.

	print(weightTensor) -- Let's have a look at our tensor before adjusting the values.

	WeightContainer:gradientDescent() -- Calling the gradientDescent() allows you to adjust the weight tensor values.

	print("\n")

	print(weightTensor) -- Let's have a look at our tensor again. This time we see changes in our values.
	
	task.wait()
	
end

```

Whew! That's quite a lot to take in, wasn't it?

That being said, you are now equipped with the fundamental knowledge on how to use this deep learning library.

Now, go have some fun with it!
