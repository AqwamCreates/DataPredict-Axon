local AqwamTensorLibrary = require(script.Parent.AqwamTensorLibraryLinker.Value)

local AutomaticDifferentiationTensor = require(script.Parent.AutomaticDifferentiationTensor)

local WeightContainer = require(script.Parent.WeightContainer)

local ActivationLayers = require(script.Parent.ActivationLayers)

local RecurrentModels = {}

local function calculateZTensor(inputTensor, inputWeightTensor, hiddenStateTensor, hiddenWeightTensor, biasTensor)
	
	return inputTensor:dotProduct{inputWeightTensor} + hiddenStateTensor:dotProduct{hiddenWeightTensor} + biasTensor
	
end

function RecurrentModels.RecurrentNeuralNetworkCell(parameterDictionary)
	
	local inputSize = parameterDictionary.inputSize or parameterDictionary[1]
	
	local hiddenSize = parameterDictionary.hiddenSize or parameterDictionary[2]
	
	local activationFunction = parameterDictionary.activationFunction or parameterDictionary[3] or "FastTanh"
	
	local learningRate = parameterDictionary.learningRate or parameterDictionary[4] or 0.001
	
	local inputWeightTensor = AutomaticDifferentiationTensor.createTensor{{inputSize, hiddenSize}}
	
	local hiddenWeightTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize, hiddenSize}}
	
	local biasTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}
	
	local WeightContainer = WeightContainer.new{}
	
	WeightContainer:setWeightTensorDataArray{
		
		{inputWeightTensor, learningRate},
		
		{hiddenWeightTensor, learningRate},
		
		{biasTensor, learningRate},
		
	}
	
	local activationLayer = ActivationLayers[activationFunction]
	
	if (not activationLayer) then error("The activation function does not exist.") end
	
	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}

	local function Model(parameterDictionary)
		
		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]
		
		local zTensor = calculateZTensor(inputTensor, inputWeightTensor, hiddenStateTensor, hiddenWeightTensor, biasTensor)
		
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

function RecurrentModels.GatedRecurrentUnitCell(parameterDictionary)

	local inputSize = parameterDictionary.inputSize or parameterDictionary[1]
	
	local hiddenSize = parameterDictionary.hiddenSize or parameterDictionary[2]
	
	local activationFunctionName = parameterDictionary.activationFunction or parameterDictionary[3] or "FastTanh"
	
	local gateActivationFunctionName = parameterDictionary.gateActivationFunction or parameterDictionary[4] or "FastSigmoid"
	
	local learningRate = parameterDictionary.learningRate or parameterDictionary[5] or 0.001

	local updateGateInputWeightTensor = AutomaticDifferentiationTensor.createTensor{{inputSize, hiddenSize}}
	
	local updateGateHiddenWeightTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize, hiddenSize}}
	
	local updateGateBiasTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}

	local resetGateInputWeightTensor = AutomaticDifferentiationTensor.createTensor{{inputSize, hiddenSize}}
	
	local resetGateHiddenWeightTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize, hiddenSize}}
	
	local resetGateBiasTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}

	local candidateInputWeightTensor = AutomaticDifferentiationTensor.createTensor{{inputSize, hiddenSize}}
	
	local candidateHiddenWeightTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize, hiddenSize}}
	
	local candidateBiasTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}

	local WeightContainer = WeightContainer.new{}

	WeightContainer:setWeightTensorDataArray{

		{updateGateInputWeightTensor, learningRate},
		
		{updateGateHiddenWeightTensor, learningRate},
		
		{updateGateBiasTensor, learningRate},

		{resetGateInputWeightTensor, learningRate},
		
		{resetGateHiddenWeightTensor, learningRate},
		
		{resetGateBiasTensor, learningRate},

		{candidateInputWeightTensor, learningRate},
		
		{candidateHiddenWeightTensor, learningRate},
		
		{candidateBiasTensor, learningRate},

	}

	local activationLayer = ActivationLayers[activationFunctionName]
	
	local gateActivationLayer = ActivationLayers[gateActivationFunctionName]
	
	if (not activationLayer) then error("The activation function does not exist.") end
	
	if (not gateActivationLayer) then error("The gate activation function does not exist.") end

	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}

	local function Model(parameterDictionary)

		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]
		
		local updateGateZTensor = calculateZTensor(inputTensor, updateGateInputWeightTensor, hiddenStateTensor, updateGateHiddenWeightTensor, updateGateBiasTensor)

		local updateGateTensor = gateActivationLayer{updateGateZTensor}
		
		local resetGateZTensor = calculateZTensor(inputTensor, resetGateInputWeightTensor, hiddenStateTensor, resetGateHiddenWeightTensor, resetGateBiasTensor)

		local resetGateTensor = gateActivationLayer{resetGateZTensor}
		
		local candidateZTensor = calculateZTensor(inputTensor, candidateInputWeightTensor, (hiddenStateTensor * resetGateTensor), candidateHiddenWeightTensor, candidateBiasTensor)

		local candidateActivationTensor = activationLayer{candidateZTensor}
		
		local oneTensorDimensionSizeArray = updateGateTensor:getDimensionSizeArray()
		
		local oneTensor = AutomaticDifferentiationTensor.createTensor{oneTensorDimensionSizeArray}

		local oneMinusUpdateGateTensor = oneTensor - updateGateTensor

		hiddenStateTensor = oneMinusUpdateGateTensor:multiply{hiddenStateTensor} + updateGateTensor:multiply{candidateActivationTensor}

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
