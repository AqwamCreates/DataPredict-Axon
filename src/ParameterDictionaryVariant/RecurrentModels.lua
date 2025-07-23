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

local defaultLearningRate = 0.0001

local defaultReverse = false

local function calculateZTensor(inputTensor, inputWeightTensor, hiddenStateTensor, hiddenWeightTensor, biasTensor)
	
	return inputTensor:dotProduct{inputWeightTensor} + hiddenStateTensor:dotProduct{hiddenWeightTensor} + biasTensor
	
end

function RecurrentModels.RecurrentNeuralNetworkCell(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
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
	
	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenSize}}

	local function Model(parameterDictionary)
		
		parameterDictionary = parameterDictionary or {}
		
		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]
		
		inputTensor = AutomaticDifferentiationTensor.coerce{inputTensor}
		
		local zTensor = calculateZTensor(inputTensor, inputWeightTensor, hiddenStateTensor, hiddenWeightTensor, biasTensor)
		
		hiddenStateTensor = activationLayer{zTensor}
		
		return hiddenStateTensor
		
	end
	
	local function reset()
		
		hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenSize}}
		
	end
	
	local function setHiddenStateTensor(parameterDictionary)
		
		hiddenStateTensor = parameterDictionary.hiddenStateTensor or parameterDictionary[1]
		
	end

	return Model, WeightContainer, reset, setHiddenStateTensor
	
end

function RecurrentModels.GatedRecurrentUnitCell(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local inputSize = parameterDictionary.inputSize or parameterDictionary[1]
	
	local hiddenSize = parameterDictionary.hiddenSize or parameterDictionary[2]
	
	local learningRate = parameterDictionary.learningRate or parameterDictionary[3] or defaultLearningRate

	local updateGateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local updateGateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local updateGateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{1, hiddenSize}}

	local resetGateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local resetGateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local resetGateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{1, hiddenSize}}

	local candidateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local candidateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local candidateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{1, hiddenSize}}

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

	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenSize}}

	local function Model(parameterDictionary)
		
		parameterDictionary = parameterDictionary or {}

		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]
		
		inputTensor = AutomaticDifferentiationTensor.coerce{inputTensor}
		
		local updateGateZTensor = calculateZTensor(inputTensor, updateGateInputWeightTensor, hiddenStateTensor, updateGateHiddenWeightTensor, updateGateBiasTensor)

		local updateGateTensor = ActivationLayers.FastSigmoid{updateGateZTensor}
		
		local resetGateZTensor = calculateZTensor(inputTensor, resetGateInputWeightTensor, hiddenStateTensor, resetGateHiddenWeightTensor, resetGateBiasTensor)

		local resetGateTensor = ActivationLayers.FastTanh{resetGateZTensor}
		
		local candidateZTensor = calculateZTensor(inputTensor, candidateInputWeightTensor, (resetGateTensor * hiddenStateTensor), candidateHiddenWeightTensor, candidateBiasTensor)

		local candidateActivationTensor = ActivationLayers.FastTanh{candidateZTensor}
		
		local oneTensorDimensionSizeArray = updateGateTensor:getDimensionSizeArray()
		
		local oneTensor = AutomaticDifferentiationTensor.createTensor{oneTensorDimensionSizeArray, 1}

		local oneMinusUpdateGateTensor = oneTensor - updateGateTensor

		hiddenStateTensor = (oneMinusUpdateGateTensor * hiddenStateTensor) + (updateGateTensor * candidateActivationTensor)

		return hiddenStateTensor

	end

	local function reset()
		
		hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenSize}}
		
	end

	local function setHiddenStateTensor(parameterDictionary)
		
		hiddenStateTensor = parameterDictionary.hiddenStateTensor or parameterDictionary[1]
		
	end

	return Model, WeightContainer, reset, setHiddenStateTensor

end

function RecurrentModels.MinimalGatedUnitCell(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local inputSize = parameterDictionary.inputSize or parameterDictionary[1]

	local hiddenSize = parameterDictionary.hiddenSize or parameterDictionary[2]

	local learningRate = parameterDictionary.learningRate or parameterDictionary[3] or defaultLearningRate

	local forgetGateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}

	local forgetGateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}

	local forgetGateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{1, hiddenSize}}

	local candidateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}

	local candidateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}

	local candidateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{1, hiddenSize}}

	local WeightContainer = WeightContainer.new{}

	WeightContainer:setWeightTensorDataArray{

		{forgetGateInputWeightTensor, learningRate},

		{forgetGateHiddenWeightTensor, learningRate},

		{forgetGateBiasTensor, learningRate},

		{candidateInputWeightTensor, learningRate},

		{candidateHiddenWeightTensor, learningRate},

		{candidateBiasTensor, learningRate},

	}

	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenSize}}

	local function Model(parameterDictionary)
		
		parameterDictionary = parameterDictionary or {}

		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]

		inputTensor = AutomaticDifferentiationTensor.coerce{inputTensor}

		local forgetGateZTensor = calculateZTensor(inputTensor, forgetGateInputWeightTensor, hiddenStateTensor, forgetGateHiddenWeightTensor, forgetGateBiasTensor)

		local forgetGateTensor = ActivationLayers.FastSigmoid{forgetGateZTensor}

		local candidateZTensor = calculateZTensor(inputTensor, candidateInputWeightTensor, (forgetGateTensor * hiddenStateTensor), candidateHiddenWeightTensor, candidateBiasTensor)

		local candidateActivationTensor = ActivationLayers.FastTanh{candidateZTensor}

		local oneTensorDimensionSizeArray = forgetGateTensor:getDimensionSizeArray()

		local oneTensor = AutomaticDifferentiationTensor.createTensor{oneTensorDimensionSizeArray, 1}

		local oneMinusUpdateGateTensor = oneTensor - forgetGateTensor

		hiddenStateTensor = (oneMinusUpdateGateTensor * hiddenStateTensor) + (forgetGateTensor * candidateActivationTensor)

		return hiddenStateTensor

	end

	local function reset()

		hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenSize}}

	end

	local function setHiddenStateTensor(parameterDictionary)

		hiddenStateTensor = parameterDictionary.hiddenStateTensor or parameterDictionary[1]

	end

	return Model, WeightContainer, reset, setHiddenStateTensor

end

function RecurrentModels.LongShortTermMemoryCell(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local inputSize = parameterDictionary.inputSize or parameterDictionary[1]
	
	local hiddenSize = parameterDictionary.hiddenSize or parameterDictionary[2]
	
	local learningRate = parameterDictionary.learningRate or parameterDictionary[3] or 0.001

	local inputGateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local inputGateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local inputGateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{1, hiddenSize}}

	local forgetGateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local forgetGateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local forgetGateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{1, hiddenSize}}

	local outputGateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local outputGateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local outputGateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{1, hiddenSize}}

	local candidateInputWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{inputSize, hiddenSize}}
	
	local candidateHiddenWeightTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{hiddenSize, hiddenSize}}
	
	local candidateBiasTensor = AutomaticDifferentiationTensor.createRandomUniformTensor{{1, hiddenSize}}

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

	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenSize}}
	
	local cellStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenSize}}

	local function Model(parameterDictionary)
		
		parameterDictionary = parameterDictionary or {}

		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]
		
		inputTensor = AutomaticDifferentiationTensor.coerce{inputTensor}

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
		
		hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenSize}}
		
		cellStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenSize}}
		
	end

	local function setHiddenStateTensor(parameterDictionary)
		
		hiddenStateTensor = parameterDictionary.hiddenStateTensor or parameterDictionary[1]
		
	end

	local function setCellStateTensor(parameterDictionary)
		
		cellStateTensor = parameterDictionary.cellStateTensor or parameterDictionary[1]
		
	end

	return Model, WeightContainer, reset, setHiddenStateTensor, setCellStateTensor

end

function RecurrentModels.UncellModel(parameterDictionary)
	
	local Model = parameterDictionary.Model or parameterDictionary[1]
	
	local reverse = parameterDictionary.reverse or parameterDictionary[2]
	
	if (not Model)then error("No model.") end
	
	local function ModifiedModel(parameterDictionary)
		
		parameterDictionary = parameterDictionary or {}
		
		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]
		
		inputTensor = AutomaticDifferentiationTensor.coerce{inputTensor}
		
		local dimensionSizeArray = inputTensor:getDimensionSizeArray()
		
		local numberOfData = dimensionSizeArray[1]
		
		local sequenceLength = dimensionSizeArray[2]
		
		local numberOfFeatures = dimensionSizeArray[3]
		
		local inputSubTensor
		
		local generatedLabelTensor
		
		local generatedLabelSubTensor
		
		if (reverse) then
			
			for sequenceIndex = sequenceLength, 1, -1 do

				inputSubTensor = inputTensor:extract{{1, sequenceIndex, 1} , {numberOfData, sequenceIndex, numberOfFeatures}}

				generatedLabelSubTensor = Model{inputSubTensor}

				if (generatedLabelTensor) then

					generatedLabelTensor = AutomaticDifferentiationTensor.concatenate{generatedLabelSubTensor, generatedLabelTensor, 2}

				else

					generatedLabelTensor = generatedLabelSubTensor

				end

			end
			
		else

			for sequenceIndex = 1, sequenceLength, 1 do

				inputSubTensor = inputTensor:extract{{1, sequenceIndex, 1} , {numberOfData, sequenceIndex, numberOfFeatures}}

				generatedLabelSubTensor = Model{inputSubTensor}

				if (generatedLabelTensor) then

					generatedLabelTensor = AutomaticDifferentiationTensor.concatenate{generatedLabelTensor, generatedLabelSubTensor, 2}

				else

					generatedLabelTensor = generatedLabelSubTensor

				end

			end
			
		end
		
		return generatedLabelTensor
		
	end
	
	return ModifiedModel
	
end

return RecurrentModels
