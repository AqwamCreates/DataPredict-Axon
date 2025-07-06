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

local AqwamTensorLibraryLinker = require(script.Parent.AqwamTensorLibraryLinker.Value)

local AutomaticDifferentiationTensor = require(script.Parent.AutomaticDifferentiationTensor)

local CostFunctions = require(script.Parent.CostFunctions)

local ReinforcementLearningModels = {}

local defaultDiscountFactor = 0.95

local function calculateCategoricalProbability(valueTensor)

	local highestActionValue = valueTensor:findMaximumValue()

	local subtractedZTensor = valueTensor - highestActionValue

	local exponentValueTensor = AutomaticDifferentiationTensor.exponent{subtractedZTensor}

	local exponentValueSumTensor = exponentValueTensor:sum{2}

	local targetActionTensor = exponentValueTensor / exponentValueSumTensor

	return targetActionTensor

end

local function calculateDiagonalGaussianProbability(meanTensor, standardDeviationTensor, noiseTensor)

	local valueTensor = meanTensor + (standardDeviationTensor * noiseTensor)

	local zScoreTensor = (valueTensor - meanTensor) / standardDeviationTensor

	local squaredZScoreTensor = zScoreTensor:power{2}

	local logValueTensorPart1 = AutomaticDifferentiationTensor.logarithm{standardDeviationTensor}

	local logValueTensorPart2 = 2 * logValueTensorPart1

	local logValueTensor = squaredZScoreTensor + logValueTensorPart2 + math.log(2 * math.pi)

	return logValueTensor

end

local function calculateRewardToGo(rewardValueHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardValueHistory, 1, -1 do

		discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function ReinforcementLearningModels.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local self = setmetatable({}, { __index = ReinforcementLearningModels })

	self.categoricalUpdateFunction = parameterDictionary.categoricalUpdateFunction or parameterDictionary[1]

	self.diagonalGaussianUpdateFunction = parameterDictionary.diagonalGaussianUpdateFunction or parameterDictionary[2]

	self.episodeUpdateFunction = parameterDictionary.episodeUpdateFunction or parameterDictionary[3]

	return self

end

function ReinforcementLearningModels.DeepQLearning(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local Model = parameterDictionary.Model or parameterDictionary[1]
	
	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]
	
	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[3] or defaultDiscountFactor
	
	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local previousQValueTensor = Model{previousFeatureTensor}
		
		local currentQValueTensor = Model{currentFeatureTensor}
		
		local maxQValue = currentQValueTensor:findMaximumValue()
		
		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue)
		
		local lastValue = previousQValueTensor[1][actionIndex]
		
		local cost = CostFunctions.FastMeanSquaredError{targetValue, previousQValueTensor}
		
		cost:differentiate()

		WeightContainer:gradientAscent()
		
		return cost
		
	end
	
	return ReinforcementLearningModels.new{categoricalUpdateFunction}
	
end

function ReinforcementLearningModels.DeepStateActionRewardStateAction(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[3] or defaultDiscountFactor

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentStateTensor, terminalStateValue)

		local previousQValueTensor = Model{previousFeatureTensor}

		local currentQValueTensor = Model{currentStateTensor}

		local targetQValueTensor = rewardValue + (discountFactor * (1 - terminalStateValue) * currentQValueTensor)

		local lastValue = previousQValueTensor[1][actionIndex]

		local costTensor = CostFunctions.FastMeanSquaredError{targetQValueTensor, previousQValueTensor}

		costTensor:differentiate()

		WeightContainer:gradientAscent()
		
		return costTensor

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction}

end

function ReinforcementLearningModels.REINFORCE(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[3] or defaultDiscountFactor
	
	local featureTensorArray = {}

	local actionProbabilityTensorArray = {}

	local rewardValueArray = {}

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)

		local actionTensor = Model{previousFeatureTensor}
		
		local actionProbabilityTensor = calculateCategoricalProbability(actionTensor)
		
		local logActionProbabilityTensor = AutomaticDifferentiationTensor.logarithm{actionProbabilityTensor}

		table.insert(featureTensorArray, previousFeatureTensor)

		table.insert(actionProbabilityTensorArray, logActionProbabilityTensor)

		table.insert(rewardValueArray, rewardValue)

		return nil

	end
	
	local episodeUpdateFunction = function(terminalStateValue)

		local rewardToGoArray = calculateRewardToGo(rewardValueArray, discountFactor)

		for h, actionProbabilityTensor in ipairs(actionProbabilityTensorArray) do

			local costTensor = actionProbabilityTensor * rewardToGoArray[h]

			costTensor:differentiate()

			WeightContainer:gradientAscent()

		end

		table.clear(featureTensorArray)

		table.clear(actionProbabilityTensorArray)

		table.clear(rewardValueArray)

		return nil

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction, nil, episodeUpdateFunction}

end

function ReinforcementLearningModels:categoricalUpdate(stateParameterDictionary)
	
	local categoricalUpdateFunction = self.categoricalUpdateFunction
	
	if (not categoricalUpdateFunction) then
		
		error("This reinforcement learning model does not support categorical updates.")
		
	end
	
	stateParameterDictionary = stateParameterDictionary or {}
	
	local previousFeatureTensor = stateParameterDictionary.previousFeatureTensor or stateParameterDictionary[1]

	local actionIndex = stateParameterDictionary.actionIndex or stateParameterDictionary[2]

	local rewardValue = stateParameterDictionary.rewardValue or stateParameterDictionary[3]

	local currentFeatureTensor = stateParameterDictionary.currentFeatureTensor or stateParameterDictionary[4]
	
	local terminalStateValue = stateParameterDictionary.terminalStateValue or stateParameterDictionary[5]
	
	return categoricalUpdateFunction(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)
	
end

function ReinforcementLearningModels:diagonalGaussianUpdate(stateParameterDictionary)

	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction

	if (not diagonalGaussianUpdateFunction) then

		error("This reinforcement learning model does not support diagonal gaussian updates.")

	end

	stateParameterDictionary = stateParameterDictionary or {}

	local previousFeatureTensor = stateParameterDictionary.previousFeatureTensor or stateParameterDictionary[1]

	local actionMeanTensor = stateParameterDictionary.actionMeanTensor or stateParameterDictionary[2]
	
	local actionStandardDeviationTensor = stateParameterDictionary.actionStandardDeviationTensor or stateParameterDictionary[3]
	
	local actionNoiseTensor = stateParameterDictionary.actionNoiseTensor or stateParameterDictionary[4]

	local rewardValue = stateParameterDictionary.rewardValue or stateParameterDictionary[5]

	local currentFeatureTensor = stateParameterDictionary.currentFeatureTensor or stateParameterDictionary[6]

	local terminalStateValue = stateParameterDictionary.terminalStateValue or stateParameterDictionary[7]

	return diagonalGaussianUpdateFunction(previousFeatureTensor, actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

end

function ReinforcementLearningModels:episodeUpdate(stateParameterDictionary)
	
	local episodeUpdateFunction = self.episodeUpdateFunction
	
	if (not episodeUpdateFunction) then return end
	
	stateParameterDictionary = stateParameterDictionary or {}
	
	local terminalStateValue = stateParameterDictionary.terminalStateValue or stateParameterDictionary[1] or 1
	
	return episodeUpdateFunction(terminalStateValue)
	
end

return ReinforcementLearningModels