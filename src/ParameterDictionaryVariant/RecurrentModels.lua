local AqwamTensorLibrary = require(script.Parent.AqwamTensorLibraryLinker.Value)

local AutomaticDifferentiationTensor = require(script.Parent.AutomaticDifferentiationTensor)

local WeightContainer = require(script.Parent.WeightContainer)

local ActivationLayers = require(script.Parent.ActivationLayers)

local RecurrentModels = {}

function RecurrentModels.RNNCell(parameterDictionary)
	
	local inputSize = parameterDictionary.inputSize or parameterDictionary[1]
	
	local hiddenSize = parameterDictionary.hiddenSize or parameterDictionary[2]
	
	local activationFunction = parameterDictionary.activationFunction or parameterDictionary[3] or "Tanh"
	
	local learningRate = parameterDictionary.learningRate or parameterDictionary[4] or 0.001
	
	local inputWeightTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize, inputSize}}
	
	local hiddenWeightTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize, hiddenSize}}
	
	local biasTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}
	
	local WeightContainer = WeightContainer.new{}
	
	WeightContainer:setWeightTensorDataArray{
		
		{inputWeightTensor, learningRate},
		
		{hiddenWeightTensor, learningRate},
		
		{biasTensor, learningRate},
		
	}
	
	local activationLayerToCreate = ActivationLayers[activationFunction]
	
	if (not activationLayerToCreate) then error("The activation function does not exist.") end
	
	local activationLayer = activationLayerToCreate()
	
	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}

	local function Model(parameterDictionary)
		
		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]
		
		local weightedInputTensor = inputWeightTensor:dotProduct{inputTensor}
		
		local weightedHiddenTensor = hiddenWeightTensor:dotProduct{hiddenStateTensor}
		
		local zTensor = weightedInputTensor + weightedHiddenTensor + biasTensor
		
		hiddenStateTensor = activationLayer{zTensor}
		
		return hiddenStateTensor
		
	end
	
	local function reset()
		
		hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}
		
	end
	
	local function setHiddenStateTensor(parameterDictionary)
		
		hiddenStateTensor = parameterDictionary.hiddenStateTensor or parameterDictionary[1]
		
	end

	return Model, WeightContainer, reset, setHiddenStateTensor
	
end

return RecurrentModels
