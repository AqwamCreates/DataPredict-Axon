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

local defaultInitializationMode = "XavierUniform"

local defaultReverse = false

local function calculateZTensor(inputTensor, inputWeightTensor, hiddenStateTensor, hiddenWeightTensor, biasTensor)

	return inputTensor:dotProduct{inputWeightTensor} + hiddenStateTensor:dotProduct{hiddenWeightTensor} + biasTensor

end

local function getCreateTensorFunction(initializationMode)
	
	local createTensorFunctionName = "create" .. initializationMode .. "Tensor"

	local createTensorFunction = AutomaticDifferentiationTensor[createTensorFunctionName]

	if (not createTensorFunction) then error("Invalid initialization mode.") end
	
	return createTensorFunction
	
end

function RecurrentModels.RecurrentNeuralNetworkCell(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local inputDimensionSize = parameterDictionary.inputDimensionSize or parameterDictionary[1]

	local hiddenDimensionSize = parameterDictionary.hiddenDimensionSize or parameterDictionary[2]

	local learningRate = parameterDictionary.learningRate or parameterDictionary[3] or defaultLearningRate

	local activationFunction = parameterDictionary.activationFunction or parameterDictionary[4] or "FastTanh"
	
	local initializationMode = parameterDictionary.initializationMode or parameterDictionary[5] or defaultInitializationMode
	
	local createTensorFunction = getCreateTensorFunction(initializationMode)

	local inputWeightTensor = parameterDictionary.inputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local hiddenWeightTensor = parameterDictionary.hiddenWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local biasTensor = parameterDictionary.biasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local WeightContainer = WeightContainer.new{}

	WeightContainer:setWeightTensorDataArray{

		{inputWeightTensor, learningRate},

		{hiddenWeightTensor, learningRate},

		{biasTensor, learningRate},

	}

	local activationLayer = ActivationLayers[activationFunction]

	if (not activationLayer) then error("The activation function does not exist.") end

	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local function Model(parameterDictionary)

		parameterDictionary = parameterDictionary or {}

		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]

		inputTensor = AutomaticDifferentiationTensor.coerce{inputTensor}

		local zTensor = calculateZTensor(inputTensor, inputWeightTensor, hiddenStateTensor, hiddenWeightTensor, biasTensor)

		hiddenStateTensor = activationLayer{zTensor}

		return hiddenStateTensor

	end

	local function reset()

		hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	end

	local function setHiddenStateTensor(parameterDictionary)

		hiddenStateTensor = parameterDictionary.hiddenStateTensor or parameterDictionary[1]

	end

	return Model, WeightContainer, reset, setHiddenStateTensor

end

function RecurrentModels.GatedRecurrentUnitCell(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local inputDimensionSize = parameterDictionary.inputDimensionSize or parameterDictionary[1]

	local hiddenDimensionSize = parameterDictionary.hiddenDimensionSize or parameterDictionary[2]

	local learningRate = parameterDictionary.learningRate or parameterDictionary[3] or defaultLearningRate
	
	local initializationMode = parameterDictionary.initializationMode or parameterDictionary[4] or defaultInitializationMode
	
	local createTensorFunction = getCreateTensorFunction(initializationMode)

	local updateGateInputWeightTensor = parameterDictionary.updateGateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local updateGateHiddenWeightTensor = parameterDictionary.updateGateHiddenWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local updateGateBiasTensor = parameterDictionary.updateGateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local resetGateInputWeightTensor = parameterDictionary.resetGateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local resetGateHiddenWeightTensor = parameterDictionary.resetGateHiddenWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local resetGateBiasTensor = parameterDictionary.resetGateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local candidateInputWeightTensor = parameterDictionary.candidateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local candidateHiddenWeightTensor = parameterDictionary.candidateHiddenWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local candidateBiasTensor = parameterDictionary.candidateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

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

	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

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

		hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	end

	local function setHiddenStateTensor(parameterDictionary)

		hiddenStateTensor = parameterDictionary.hiddenStateTensor or parameterDictionary[1]

	end

	return Model, WeightContainer, reset, setHiddenStateTensor

end

function RecurrentModels.MinimalGatedUnitCell(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local inputDimensionSize = parameterDictionary.inputDimensionSize or parameterDictionary[1]

	local hiddenDimensionSize = parameterDictionary.hiddenDimensionSize or parameterDictionary[2]

	local learningRate = parameterDictionary.learningRate or parameterDictionary[3] or defaultLearningRate
	
	local initializationMode = parameterDictionary.initializationMode or parameterDictionary[4] or defaultInitializationMode

	local createTensorFunction = getCreateTensorFunction(initializationMode)

	local forgetGateInputWeightTensor = parameterDictionary.forgetGateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local forgetGateHiddenWeightTensor = parameterDictionary.forgetGateHiddenWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local forgetGateBiasTensor = parameterDictionary.forgetGateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local candidateInputWeightTensor = parameterDictionary.candidateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local candidateHiddenWeightTensor = parameterDictionary.candidateHiddenWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local candidateBiasTensor = parameterDictionary.candidateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local WeightContainer = WeightContainer.new{}

	WeightContainer:setWeightTensorDataArray{

		{forgetGateInputWeightTensor, learningRate},

		{forgetGateHiddenWeightTensor, learningRate},

		{forgetGateBiasTensor, learningRate},

		{candidateInputWeightTensor, learningRate},

		{candidateHiddenWeightTensor, learningRate},

		{candidateBiasTensor, learningRate},

	}

	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

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

		hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	end

	local function setHiddenStateTensor(parameterDictionary)

		hiddenStateTensor = parameterDictionary.hiddenStateTensor or parameterDictionary[1]

	end

	return Model, WeightContainer, reset, setHiddenStateTensor

end

function RecurrentModels.LightGatedRecurrentUnitCell(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local inputDimensionSize = parameterDictionary.inputDimensionSize or parameterDictionary[1]

	local hiddenDimensionSize = parameterDictionary.hiddenDimensionSize or parameterDictionary[2]

	local learningRate = parameterDictionary.learningRate or parameterDictionary[3] or defaultLearningRate
	
	local initializationMode = parameterDictionary.initializationMode or parameterDictionary[4] or defaultInitializationMode

	local createTensorFunction = getCreateTensorFunction(initializationMode)
	
	local zInputWeightTensor = parameterDictionary.zInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local zHiddenWeightTensor = parameterDictionary.zHiddenWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local hInputWeightTensor = parameterDictionary.hInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local hHiddenWeightTensor = parameterDictionary.hHiddenWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local WeightContainer = WeightContainer.new{}

	WeightContainer:setWeightTensorDataArray{

		{zInputWeightTensor, learningRate},

		{zHiddenWeightTensor, learningRate},

		{hInputWeightTensor, learningRate},

		{hHiddenWeightTensor, learningRate},

	}

	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local function Model(parameterDictionary)

		parameterDictionary = parameterDictionary or {}

		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]

		inputTensor = AutomaticDifferentiationTensor.coerce{inputTensor}
		
		local zWeightedInputTensor = inputTensor:dotProduct{zInputWeightTensor}
		
		local zWeightedHiddenTensor = hiddenStateTensor:dotProduct{zHiddenWeightTensor}
		
		local batchNormalizationZWeightedInputTensor = zWeightedInputTensor:zScoreNormalization{1}
		
		local hWeightedInputTensor = inputTensor:dotProduct{hInputWeightTensor}
		
		local hWeightedHiddenTensor = hiddenStateTensor:dotProduct{hHiddenWeightTensor}

		local batchNormalizationHWeightedInputTensor = hWeightedInputTensor:zScoreNormalization{1}
		
		local zTensor = ActivationLayers.FastSigmoid{zWeightedInputTensor + zWeightedHiddenTensor}
		
		local hTensor = ActivationLayers.FastRectifiedLinearUnit{hWeightedInputTensor + hWeightedHiddenTensor}
		
		hiddenStateTensor = (zTensor * hiddenStateTensor) + (1 - zTensor) * hTensor

		return hiddenStateTensor

	end

	local function reset()

		hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	end

	local function setHiddenStateTensor(parameterDictionary)

		hiddenStateTensor = parameterDictionary.hiddenStateTensor or parameterDictionary[1]

	end

	return Model, WeightContainer, reset, setHiddenStateTensor

end

function RecurrentModels.SimpleRecurrentUnitCell(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local inputDimensionSize = parameterDictionary.inputDimensionSize or parameterDictionary[1]

	local hiddenDimensionSize = parameterDictionary.hiddenDimensionSize or parameterDictionary[2]

	local learningRate = parameterDictionary.learningRate or parameterDictionary[3] or defaultLearningRate
	
	local initializationMode = parameterDictionary.initializationMode or parameterDictionary[4] or defaultInitializationMode

	local createTensorFunction = getCreateTensorFunction(initializationMode)
	
	local forgetGateInputWeightTensor = parameterDictionary.forgetGateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local forgetGateCellStateMultiplerTensor = parameterDictionary.forgetGateCellStateMultiplerTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local forgetGateBiasTensor = parameterDictionary.forgetGateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}
	
	local cellInputWeightTensor = parameterDictionary.cellInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local resetGateInputWeightTensor = parameterDictionary.resetGateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local resetGateCellStateMultiplerTensor = parameterDictionary.resetGateCellStateMultiplerTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local resetGateBiasTensor = parameterDictionary.resetGateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local WeightContainer = WeightContainer.new{}

	WeightContainer:setWeightTensorDataArray{

		{forgetGateInputWeightTensor, learningRate},

		{forgetGateCellStateMultiplerTensor, learningRate},

		{forgetGateBiasTensor, learningRate},

		{cellInputWeightTensor, learningRate},
		
		{resetGateInputWeightTensor, learningRate},
		
		{resetGateCellStateMultiplerTensor, learningRate},
		
		{resetGateBiasTensor, learningRate},

	}

	local cellStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local function Model(parameterDictionary)

		parameterDictionary = parameterDictionary or {}

		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]

		inputTensor = AutomaticDifferentiationTensor.coerce{inputTensor}
		
		local zForgetGateTensor = inputTensor:dotProduct{forgetGateInputWeightTensor} + (forgetGateCellStateMultiplerTensor * cellStateTensor) + forgetGateBiasTensor
		
		local forgetGateTensor = ActivationLayers.FastSigmoid{zForgetGateTensor}
		
		cellStateTensor = (forgetGateTensor * cellStateTensor) + ((1 - forgetGateTensor) * inputTensor:dotProduct{cellInputWeightTensor})
		
		local zResetGateTensor = inputTensor:dotProduct{resetGateInputWeightTensor} + (resetGateCellStateMultiplerTensor * cellStateTensor) + resetGateBiasTensor
		
		local resetGateTensor = ActivationLayers.FastSigmoid{zResetGateTensor}
		
		local hiddenStateTensor = (resetGateTensor * cellStateTensor) + ((1 - resetGateTensor) * inputTensor)

		return hiddenStateTensor

	end

	local function reset()

		cellStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	end

	local function setCellStateTensor(parameterDictionary)

		cellStateTensor = parameterDictionary.cellStateTensor or parameterDictionary[1]

	end

	return Model, WeightContainer, reset, setCellStateTensor

end

function RecurrentModels.LongShortTermMemoryCell(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local inputDimensionSize = parameterDictionary.inputDimensionSize or parameterDictionary[1]

	local hiddenDimensionSize = parameterDictionary.hiddenDimensionSize or parameterDictionary[2]

	local learningRate = parameterDictionary.learningRate or parameterDictionary[3] or defaultLearningRate
	
	local initializationMode = parameterDictionary.initializationMode or parameterDictionary[4] or defaultInitializationMode

	local createTensorFunction = getCreateTensorFunction(initializationMode)

	local inputGateInputWeightTensor = parameterDictionary.inputGateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local inputGateHiddenWeightTensor = parameterDictionary.inputGateHiddenWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local inputGateBiasTensor = parameterDictionary.inputGateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local forgetGateInputWeightTensor = parameterDictionary.forgetGateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local forgetGateHiddenWeightTensor = parameterDictionary.forgetGateHiddenWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local forgetGateBiasTensor = parameterDictionary.forgetGateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local outputGateInputWeightTensor = parameterDictionary.outputGateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local outputGateHiddenWeightTensor = parameterDictionary.outputGateHiddenWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local outputGateBiasTensor = parameterDictionary.outputGateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local cellInputWeightTensor = parameterDictionary.cellInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local cellHiddenWeightTensor = parameterDictionary.cellHiddenWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}

	local cellBiasTensor = parameterDictionary.cellBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

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

		{cellInputWeightTensor, learningRate},

		{cellHiddenWeightTensor, learningRate},

		{cellBiasTensor, learningRate},
	}

	local hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local cellStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local function Model(parameterDictionary)

		parameterDictionary = parameterDictionary or {}

		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]

		inputTensor = AutomaticDifferentiationTensor.coerce{inputTensor}

		local inputGateZTensor = calculateZTensor(inputTensor, inputGateInputWeightTensor, hiddenStateTensor, inputGateHiddenWeightTensor, inputGateBiasTensor)

		local forgetGateZTensor = calculateZTensor(inputTensor, forgetGateInputWeightTensor, hiddenStateTensor, forgetGateHiddenWeightTensor, forgetGateBiasTensor)

		local outputGateZTensor = calculateZTensor(inputTensor, outputGateInputWeightTensor, hiddenStateTensor, outputGateHiddenWeightTensor, outputGateBiasTensor)

		local cellZTensor = calculateZTensor(inputTensor, cellInputWeightTensor, hiddenStateTensor, cellHiddenWeightTensor, cellBiasTensor)

		local inputGateTensor = ActivationLayers.FastSigmoid{inputGateZTensor}

		local forgetGateTensor = ActivationLayers.FastSigmoid{forgetGateZTensor}

		local outputGateTensor = ActivationLayers.FastSigmoid{outputGateZTensor}

		local cellStateTensor = ActivationLayers.FastTanh{cellZTensor}

		cellStateTensor = (forgetGateTensor * cellStateTensor) + (inputGateTensor * cellStateTensor)

		hiddenStateTensor = outputGateTensor * ActivationLayers.FastTanh{cellStateTensor}

		return hiddenStateTensor

	end

	local function reset()

		hiddenStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

		cellStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	end

	local function setHiddenStateTensor(parameterDictionary)

		hiddenStateTensor = parameterDictionary.hiddenStateTensor or parameterDictionary[1]

	end

	local function setCellStateTensor(parameterDictionary)

		cellStateTensor = parameterDictionary.cellStateTensor or parameterDictionary[1]

	end

	return Model, WeightContainer, reset, setHiddenStateTensor, setCellStateTensor

end

function RecurrentModels.PeepholeLongShortTermMemoryCell(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local inputDimensionSize = parameterDictionary.inputDimensionSize or parameterDictionary[1]
	
	local hiddenDimensionSize = parameterDictionary.hiddenDimensionSize or parameterDictionary[2]
	
	local learningRate = parameterDictionary.learningRate or parameterDictionary[3] or defaultLearningRate
	
	local initializationMode = parameterDictionary.initializationMode or parameterDictionary[4] or defaultInitializationMode

	local createTensorFunction = getCreateTensorFunction(initializationMode)

	local inputGateInputWeightTensor = parameterDictionary.inputGateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local inputGateCandidateWeightTensor = parameterDictionary.inputGateCandidateWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local inputGateBiasTensor = parameterDictionary.inputGateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local forgetGateInputWeightTensor = parameterDictionary.forgetGateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local forgetGateCandidateWeightTensor = parameterDictionary.forgetGateCandidateWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local forgetGateBiasTensor = parameterDictionary.forgetGateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local outputGateInputWeightTensor = parameterDictionary.outputGateInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local outputGateCandidateWeightTensor = parameterDictionary.outputGateCandidateWeightTensor or createTensorFunction{{hiddenDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = hiddenDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local outputGateBiasTensor = parameterDictionary.outputGateBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local cellInputWeightTensor = parameterDictionary.cellInputWeightTensor or createTensorFunction{{inputDimensionSize, hiddenDimensionSize}, numberOfInputNeurons = inputDimensionSize, numberOfOutputNeurons = hiddenDimensionSize}
	
	local cellBiasTensor = parameterDictionary.cellBiasTensor or createTensorFunction{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local WeightContainer = WeightContainer.new{}
	
	WeightContainer:setWeightTensorDataArray{
		
		{inputGateInputWeightTensor, learningRate},
		
		{inputGateCandidateWeightTensor, learningRate},
		
		{inputGateBiasTensor, learningRate},
		
		{forgetGateInputWeightTensor, learningRate},
		
		{forgetGateCandidateWeightTensor, learningRate},
		
		{forgetGateBiasTensor, learningRate},
		
		{outputGateInputWeightTensor, learningRate},
		
		{outputGateCandidateWeightTensor, learningRate},
		
		{outputGateBiasTensor, learningRate},
		
		{cellInputWeightTensor, learningRate},
		
		{cellBiasTensor, learningRate},
		
	}

	local cellStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}

	local function Model(parameterDictionary)
		
		parameterDictionary = parameterDictionary or {}
		
		local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]
		
		inputTensor = AutomaticDifferentiationTensor.coerce{inputTensor}

		local inputGateZTensor = calculateZTensor(inputTensor, inputGateInputWeightTensor, cellStateTensor, inputGateCandidateWeightTensor, inputGateBiasTensor)
		
		local forgetGateZTensor = calculateZTensor(inputTensor, forgetGateInputWeightTensor, cellStateTensor, forgetGateCandidateWeightTensor, forgetGateBiasTensor)
		
		local outputGateZTensor = calculateZTensor(inputTensor, outputGateInputWeightTensor, cellStateTensor, outputGateCandidateWeightTensor, outputGateBiasTensor)
		
		local inputGateTensor = ActivationLayers.FastSigmoid{inputGateZTensor}

		local forgetGateTensor = ActivationLayers.FastSigmoid{forgetGateZTensor}

		local outputGateTensor = ActivationLayers.FastSigmoid{outputGateZTensor}
		
		local cellStateTensorPart1 = ActivationLayers.FastSigmoid{inputTensor:dotProduct{cellInputWeightTensor} + cellBiasTensor}
		
		cellStateTensor = (forgetGateTensor * cellStateTensor) + (inputGateTensor * cellStateTensorPart1)
			
		local activatedCellStateTensor = outputGateTensor * ActivationLayers.FastSigmoid{cellStateTensor}

		return activatedCellStateTensor
	end

	local function reset()
		
		cellStateTensor = AutomaticDifferentiationTensor.createTensor{{1, hiddenDimensionSize}, numberOfInputNeurons = 1, numberOfOutputNeurons = hiddenDimensionSize}
		
	end

	local function setCellStateTensor(parameterDictionary)
		
		cellStateTensor = parameterDictionary.cellStateTensor or parameterDictionary[1]
		
	end

	return Model, WeightContainer, reset, setCellStateTensor
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
