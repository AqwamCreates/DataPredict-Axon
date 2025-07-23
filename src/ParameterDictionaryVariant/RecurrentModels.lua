--[[

	--------------------------------------------------------------------

	Aqwam's Deep Learning Library (DataPredict Axon)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict-Axon/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require(script.Parent.AqwamTensorLibraryLinker.Value)

local AutomaticDifferentiationTensor = require(script.Parent.AutomaticDifferentiationTensor)

local WeightContainer = require(script.Parent.WeightContainer)

local ActivationLayers = require(script.Parent.ActivationLayers)

local RecurrentModels = {}

local defaultLearningRate = 0.001

local function calculateZTensor(inputTensor, inputWeightTensor, hiddenStateTensor, hiddenWeightTensor, biasTensor)
	
	return inputTensor:dotProduct{inputWeightTensor} + hiddenStateTensor:dotProduct{hiddenWeightTensor} + biasTensor
	
end

function RecurrentModels.RecurrentNeuralNetworkCell(parameterDictionary)
	
	local inputSize = parameterDictionary.inputSize or parameterDictionary[1]
	
	local hiddenSize = parameterDictionary.hiddenSize or parameterDictionary[2]
	
	local activationFunction = parameterDictionary.activationFunction or parameterDictionary[3] or "FastTanh"
	
	local learningRate = parameterDictionary.learningRate or parameterDictionary[4] or defaultLearningRate
	
	local inputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local hiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local biasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize}}
	
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
	
	local learningRate = parameterDictionary.learningRate or parameterDictionary[3] or defaultLearningRate

	local updateGateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local updateGateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local updateGateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize}}

	local resetGateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local resetGateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local resetGateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize}}

	local candidateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local candidateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local candidateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize}}

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

	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}

	local function Model(parameterDictionary)

		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]
		
		local updateGateZTensor = calculateZTensor(inputTensor, updateGateInputWeightTensor, hiddenStateTensor, updateGateHiddenWeightTensor, updateGateBiasTensor)

		local updateGateTensor = ActivationLayers.FastSigmoid{updateGateZTensor}
		
		local resetGateZTensor = calculateZTensor(inputTensor, resetGateInputWeightTensor, hiddenStateTensor, resetGateHiddenWeightTensor, resetGateBiasTensor)

		local resetGateTensor = ActivationLayers.FastTanh{resetGateZTensor}
		
		local candidateZTensor = calculateZTensor(inputTensor, candidateInputWeightTensor, (hiddenStateTensor * resetGateTensor), candidateHiddenWeightTensor, candidateBiasTensor)

		local candidateActivationTensor = ActivationLayers.FastTanh{candidateZTensor}
		
		local oneTensorDimensionSizeArray = updateGateTensor:getDimensionSizeArray()
		
		local oneTensor = AutomaticDifferentiationTensor.createTensor{oneTensorDimensionSizeArray, 1}

		local oneMinusUpdateGateTensor = oneTensor - updateGateTensor

		hiddenStateTensor = (oneMinusUpdateGateTensor * hiddenStateTensor) + (updateGateTensor * candidateActivationTensor)

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

function RecurrentModels.LongShortTermMemoryCell(parameterDictionary)

	local inputSize = parameterDictionary.inputSize or parameterDictionary[1]
	
	local hiddenSize = parameterDictionary.hiddenSize or parameterDictionary[2]
	
	local learningRate = parameterDictionary.learningRate or parameterDictionary[3] or 0.001

	local inputGateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local inputGateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local inputGateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize}}

	local forgetGateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local forgetGateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local forgetGateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize}}

	local outputGateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local outputGateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local outputGateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize}}

	local candidateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local candidateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local candidateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize}}

	local WeightContainer = WeightContainer.new{}
	
	WeightContainer:setWeightTensorDataArray{

		{inputGateInputWeightTensor, learningRate},
		
		{inputGateHiddenWeightTensor, learningRate},
		
		{inputGateBiasTensor, learningRate},

		{forgetGateInputWeightTensor, learningRate},
		
		{forgetGateHiddenWeightTensor, learningRate},
		
		{forgetGateBiasTensor, learningRate},

		{outputGateInputWeightTensor, learningRate},
		
		{outputGateHiddenWeightTensor, learningRate},
		
		{outputGateBiasTensor, learningRate},

		{candidateInputWeightTensor, learningRate},
		
		{candidateHiddenWeightTensor, learningRate},
		
		{candidateBiasTensor, learningRate},
	}

	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}
	
	local cellStateTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}

	local function Model(parameterDictionary)

		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]

		local inputGateZTensor = calculateZTensor(inputTensor, inputGateInputWeightTensor, hiddenStateTensor, inputGateHiddenWeightTensor, inputGateBiasTensor)

		local forgetGateZTensor = calculateZTensor(inputTensor, forgetGateInputWeightTensor, hiddenStateTensor, forgetGateHiddenWeightTensor, forgetGateBiasTensor)

		local outputGateZTensor = calculateZTensor(inputTensor, outputGateInputWeightTensor, hiddenStateTensor, outputGateHiddenWeightTensor, outputGateBiasTensor)

		local candidateZTensor = calculateZTensor(inputTensor, candidateInputWeightTensor, hiddenStateTensor, candidateHiddenWeightTensor, candidateBiasTensor)

		local inputGateTensor = ActivationLayers.FastSigmoid{inputGateZTensor}
		
		local forgetGateTensor = ActivationLayers.FastSigmoid{forgetGateZTensor}
		
		local outputGateTensor = ActivationLayers.FastSigmoid{outputGateZTensor}
		
		local candidateTensor = ActivationLayers.FastTanh{candidateZTensor}

		cellStateTensor = (forgetGateTensor * cellStateTensor) + (inputGateTensor * candidateTensor)

		local activatedCellStateTensor = ActivationLayers.FastTanh{cellStateTensor}
		
		hiddenStateTensor = outputGateTensor * activatedCellStateTensor

		return activatedCellStateTensor

	end

	local function reset()
		
		hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}
		
		cellStateTensor = AutomaticDifferentiationTensor.createTensor{{hiddenSize}}
		
	end

	local function setHiddenStateTensor(parameterDictionary)
		
		hiddenStateTensor = parameterDictionary.hiddenStateTensor or parameterDictionary[1]
		
	end

	local function setCellStateTensor(parameterDictionary)
		
		cellStateTensor = parameterDictionary.cellStateTensor or parameterDictionary[1]
		
	end

	return Model, WeightContainer, reset, setHiddenStateTensor, setCellStateTensor

end


return RecurrentModels
