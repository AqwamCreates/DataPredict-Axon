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

local CostFunctions = require(script.Parent.CostFunctions)

local ReinforcementLearningModels = {}

local defaultClipRatio = 0.3

local defaultEpsilon = 0.5

local defaultAlpha = 0.1

local defaultNoiseClippingFactor = 0.5

local defaultPolicyDelayAmount = 3

local defaultAveragingRate = 0.995

local defaultDiscountFactor = 0.95

local function rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetWeightTensorArray, 1 do

		local TargetWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRate, TargetWeightTensorArray[layer])

		local PrimaryWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryWeightTensorArray[layer])

		TargetWeightTensorArray[layer] = AqwamTensorLibrary:add(TargetWeightTensorArrayPart, PrimaryWeightTensorArrayPart)

	end

	return TargetWeightTensorArray

end

local function calculateCategoricalProbability(valueTensor)

	local highestActionValue = valueTensor:findMaximumValue()

	local subtractedZTensor = valueTensor - highestActionValue

	local exponentValueTensor = AutomaticDifferentiationTensor.exponent{subtractedZTensor}

	local exponentValueSumTensor = exponentValueTensor:sum{2}

	local targetActionTensor = exponentValueTensor / exponentValueSumTensor

	return exponentValueTensor

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
		
		local cost = CostFunctions.FastMeanSquaredError{targetValue, lastValue}
		
		cost:differentiate()

		WeightContainer:gradientAscent()
		
		cost:destroy{true}
		
	end
	
	return ReinforcementLearningModels.new{categoricalUpdateFunction}
	
end

function ReinforcementLearningModels.DeepDoubleQLearningV1(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[3] or defaultDiscountFactor
	
	local WeightTensorArrayArray = parameterDictionary.WeightTensorArrayArray or parameterDictionary[4] or {}
	
	WeightTensorArrayArray[1] = WeightTensorArrayArray[1] or WeightContainer:getWeightTensorArray{true} -- So that the changes are reflected to the weight tensors that are put into the WeightContainer.

	WeightTensorArrayArray[2] = WeightTensorArrayArray[2] or WeightContainer:getWeightTensorArray{} -- To ensure that a copy is made to avoid gradient contribution to the current actor weight tensors.

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local randomProbability = math.random()
		
		local updateSecondWeightTensorArray = (randomProbability >= 0.5)
		
		local selectedWeightTensorArrayNumberForTargetVector = (updateSecondWeightTensorArray and 1) or 2

		local selectedWeightTensorArrayNumberForUpdate = (updateSecondWeightTensorArray and 2) or 1
		
		WeightContainer:setWeightTensorArray{WeightTensorArrayArray[selectedWeightTensorArrayNumberForUpdate]}

		local previousQValueTensor = Model{previousFeatureTensor}
		
		WeightContainer:setWeightTensorArray{WeightTensorArrayArray[selectedWeightTensorArrayNumberForTargetVector]}

		local currentQValueTensor = Model{currentFeatureTensor}

		local maxQValue = currentQValueTensor:findMaximumValue()

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue)

		local lastValue = previousQValueTensor[1][actionIndex]

		local cost = CostFunctions.FastMeanSquaredError{targetValue, lastValue}
		
		WeightContainer:setWeightTensorArray{WeightTensorArrayArray[selectedWeightTensorArrayNumberForUpdate]}

		cost:differentiate()

		WeightContainer:gradientAscent()
		
		WeightTensorArrayArray[selectedWeightTensorArrayNumberForUpdate] = WeightContainer:getWeightTensorArray{true}

		cost:destroy{true}

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction}

end

function ReinforcementLearningModels.DeepDoubleQLearningV2(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]
	
	local averagingRate = parameterDictionary.averagingRate or parameterDictionary[3] or defaultAveragingRate

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[4] or defaultDiscountFactor

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local PrimaryWeightTensorArray = WeightContainer:getWeightTensorArray{true}

		local currentQValueTensor = Model{currentFeatureTensor}
		
		local maxQValue = currentQValueTensor:findMaximumValue()

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue)

		local previousQValueTensor = Model{previousFeatureTensor}

		local lastValue = previousQValueTensor[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue
		
		temporalDifferenceError:differentiate()
		
		WeightContainer:gradientAscent()

		local TargetWeightTensorArray = WeightContainer:getWeightTensorArray{true}

		TargetWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

		WeightContainer:setWeightTensorArray{TargetWeightTensorArray, true}

		temporalDifferenceError:destroy{true}

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction}

end

function ReinforcementLearningModels.DeepClippedDoubleQLearning(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[3] or defaultDiscountFactor

	local WeightTensorArrayArray = parameterDictionary.WeightTensorArrayArray or parameterDictionary[4] or {}

	WeightTensorArrayArray[1] = WeightTensorArrayArray[1] or WeightContainer:getWeightTensorArray{true} -- So that the changes are reflected to the weight tensors that are put into the WeightContainer.

	WeightTensorArrayArray[2] = WeightTensorArrayArray[2] or WeightContainer:getWeightTensorArray{} -- To ensure that a copy is made to avoid gradient contribution to the current actor weight tensors.

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local maxQValueArray = {}

		for i = 1, 2, 1 do

			WeightContainer:setWeightTensorArray{WeightTensorArrayArray[i], true}

			local currentQValueTensor = Model{currentFeatureTensor}
			
			local currentMaxQValue = currentQValueTensor:findMaximumValue()

			table.insert(maxQValueArray, currentMaxQValue)

		end
		
		local minimumCurrentMaxQValue = AutomaticDifferentiationTensor.minimum(maxQValueArray)
		
		for i = 1, 2, 1 do

			WeightContainer:setWeightTensorArray{WeightTensorArrayArray[i], true}

			local previousQValueTensor = Model{previousFeatureTensor}

			local previousQValue = previousQValueTensor[1][actionIndex]

			local cost = CostFunctions.FastMeanSquaredError{minimumCurrentMaxQValue, previousQValue}
			
			cost:differentiate()
			
			WeightContainer:gradientAscent()

		end

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

		local costTensor = CostFunctions.FastMeanSquaredError{targetQValueTensor, previousQValueTensor}

		costTensor:differentiate()

		WeightContainer:gradientAscent()
		
		costTensor:destroy{true}

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction}

end

function ReinforcementLearningModels.DeepDoubleStateActionRewardStateActionV1(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[3] or defaultDiscountFactor

	local WeightTensorArrayArray = parameterDictionary.WeightTensorArrayArray or parameterDictionary[4] or {}

	WeightTensorArrayArray[1] = WeightTensorArrayArray[1] or WeightContainer:getWeightTensorArray{true} -- So that the changes are reflected to the weight tensors that are put into the WeightContainer.

	WeightTensorArrayArray[2] = WeightTensorArrayArray[2] or WeightContainer:getWeightTensorArray{} -- To ensure that a copy is made to avoid gradient contribution to the current actor weight tensors.

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)

		local randomProbability = math.random()

		local updateSecondWeightTensorArray = (randomProbability >= 0.5)

		local selectedWeightTensorArrayNumberForTargetVector = (updateSecondWeightTensorArray and 1) or 2

		local selectedWeightTensorArrayNumberForUpdate = (updateSecondWeightTensorArray and 2) or 1

		WeightContainer:setWeightTensorArray{WeightTensorArrayArray[selectedWeightTensorArrayNumberForUpdate]}

		local previousQValueTensor = Model{previousFeatureTensor}

		WeightContainer:setWeightTensorArray{WeightTensorArrayArray[selectedWeightTensorArrayNumberForTargetVector]}

		local currentQValueTensor = Model{currentFeatureTensor}

		local targetQValueTensor = rewardValue + (discountFactor * (1 - terminalStateValue) * currentQValueTensor)

		local costTensor = CostFunctions.FastMeanSquaredError{targetQValueTensor, previousQValueTensor}

		WeightContainer:setWeightTensorArray{WeightTensorArrayArray[selectedWeightTensorArrayNumberForUpdate]}

		costTensor:differentiate()

		WeightContainer:gradientAscent()

		WeightTensorArrayArray[selectedWeightTensorArrayNumberForUpdate] = WeightContainer:getWeightTensorArray{true}

		costTensor:destroy{true}

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction}

end

function ReinforcementLearningModels.DeepDoubleStateActionRewardStateActionV2(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]

	local averagingRate = parameterDictionary.averagingRate or parameterDictionary[3] or defaultAveragingRate

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[4] or defaultDiscountFactor

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)

		local PrimaryWeightTensorArray = WeightContainer:getWeightTensorArray{true}

		local currentQValueTensor = Model{currentFeatureTensor}

		local previousQValueTensor = Model{previousFeatureTensor}
		
		local targetQValueTensor = rewardValue + (discountFactor * (1 - terminalStateValue) * currentQValueTensor)

		local costTensor = CostFunctions.FastMeanSquaredError{targetQValueTensor, previousQValueTensor}

		costTensor:differentiate()

		WeightContainer:gradientAscent()

		local TargetWeightTensorArray = WeightContainer:getWeightTensorArray{true}

		TargetWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

		WeightContainer:setWeightTensorArray{TargetWeightTensorArray, true}

		costTensor:destroy{true}

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction}

end

function ReinforcementLearningModels.DeepExpectedStateActionRewardStateAction(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]
	
	local epsilon = parameterDictionary.epsilon or parameterDictionary[3] or defaultEpsilon

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[4] or defaultDiscountFactor

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local numberOfGreedyActions = 0
		
		local expectedQValue = 0

		local previousQValueTensor = Model{previousFeatureTensor}

		local currentQValueTensor = Model{currentFeatureTensor}

		local maxQValue = currentQValueTensor:findMaximumValue()
		
		local unwrappedTargetTensor = currentQValueTensor[1]
		
		local numberOfClasses = #unwrappedTargetTensor

		for i = 1, numberOfClasses, 1 do

			if (unwrappedTargetTensor[i] == maxQValue) then
				
				numberOfGreedyActions = numberOfGreedyActions + 1
				
			end

		end

		local nonGreedyActionProbability = epsilon / numberOfClasses

		local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		for _, qValue in ipairs(unwrappedTargetTensor) do

			if (qValue == maxQValue) then

				expectedQValue = expectedQValue + (qValue * greedyActionProbability)

			else

				expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

			end

		end

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * expectedQValue)

		local lastValue = previousQValueTensor[1][actionIndex]

		local cost = CostFunctions.FastMeanSquaredError{targetValue, lastValue}

		cost:differentiate()

		WeightContainer:gradientAscent()

		cost:destroy{true}

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction}

end

function ReinforcementLearningModels.DeepDoubleExpectedStateActionRewardStateActionV1(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]
	
	local epsilon = parameterDictionary.epsilon or parameterDictionary[3] or defaultEpsilon

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[4] or defaultDiscountFactor

	local WeightTensorArrayArray = parameterDictionary.WeightTensorArrayArray or parameterDictionary[5] or {}

	WeightTensorArrayArray[1] = WeightTensorArrayArray[1] or WeightContainer:getWeightTensorArray{true} -- So that the changes are reflected to the weight tensors that are put into the WeightContainer.

	WeightTensorArrayArray[2] = WeightTensorArrayArray[2] or WeightContainer:getWeightTensorArray{} -- To ensure that a copy is made to avoid gradient contribution to the current actor weight tensors.

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local numberOfGreedyActions = 0

		local expectedQValue = 0

		local randomProbability = math.random()

		local updateSecondWeightTensorArray = (randomProbability >= 0.5)

		local selectedWeightTensorArrayNumberForTargetVector = (updateSecondWeightTensorArray and 1) or 2

		local selectedWeightTensorArrayNumberForUpdate = (updateSecondWeightTensorArray and 2) or 1

		WeightContainer:setWeightTensorArray{WeightTensorArrayArray[selectedWeightTensorArrayNumberForUpdate]}

		local previousQValueTensor = Model{previousFeatureTensor}

		WeightContainer:setWeightTensorArray{WeightTensorArrayArray[selectedWeightTensorArrayNumberForTargetVector]}

		local currentQValueTensor = Model{currentFeatureTensor}

		local maxQValue = currentQValueTensor:findMaximumValue()

		local unwrappedTargetTensor = currentQValueTensor[1]

		local numberOfClasses = #unwrappedTargetTensor

		for i = 1, numberOfClasses, 1 do

			if (unwrappedTargetTensor[i] == maxQValue) then

				numberOfGreedyActions = numberOfGreedyActions + 1

			end

		end

		local nonGreedyActionProbability = epsilon / numberOfClasses

		local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		for _, qValue in ipairs(unwrappedTargetTensor) do

			if (qValue == maxQValue) then

				expectedQValue = expectedQValue + (qValue * greedyActionProbability)

			else

				expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

			end

		end

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * expectedQValue)

		local lastValue = previousQValueTensor[1][actionIndex]

		local cost = CostFunctions.FastMeanSquaredError{targetValue, lastValue}

		WeightContainer:setWeightTensorArray{WeightTensorArrayArray[selectedWeightTensorArrayNumberForUpdate]}

		cost:differentiate()

		WeightContainer:gradientAscent()

		WeightTensorArrayArray[selectedWeightTensorArrayNumberForUpdate] = WeightContainer:getWeightTensorArray{true}

		cost:destroy{true}

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction}

end

function ReinforcementLearningModels.DeepDoubleExpectedStateActionRewardStateActionV2(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]
	
	local epsilon = parameterDictionary.epsilon or parameterDictionary[3] or defaultEpsilon

	local averagingRate = parameterDictionary.averagingRate or parameterDictionary[4] or defaultAveragingRate

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[5] or defaultDiscountFactor

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local numberOfGreedyActions = 0

		local expectedQValue = 0

		local PrimaryWeightTensorArray = WeightContainer:getWeightTensorArray{true}

		local currentQValueTensor = Model{currentFeatureTensor}

		local previousQValueTensor = Model{previousFeatureTensor}

		local maxQValue = currentQValueTensor:findMaximumValue()

		local unwrappedTargetTensor = currentQValueTensor[1]

		local numberOfClasses = #unwrappedTargetTensor

		for i = 1, numberOfClasses, 1 do

			if (unwrappedTargetTensor[i] == maxQValue) then

				numberOfGreedyActions = numberOfGreedyActions + 1

			end

		end

		local nonGreedyActionProbability = epsilon / numberOfClasses

		local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		for _, qValue in ipairs(unwrappedTargetTensor) do

			if (qValue == maxQValue) then

				expectedQValue = expectedQValue + (qValue * greedyActionProbability)

			else

				expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

			end

		end

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * expectedQValue)

		local lastValue = previousQValueTensor[1][actionIndex]

		local cost = CostFunctions.FastMeanSquaredError{targetValue, lastValue}

		cost:differentiate()

		WeightContainer:gradientAscent()

		local TargetWeightTensorArray = WeightContainer:getWeightTensorArray{true}

		TargetWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

		WeightContainer:setWeightTensorArray{TargetWeightTensorArray, true}

		cost:destroy{true}

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction}

end

function ReinforcementLearningModels.REINFORCE(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[3] or defaultDiscountFactor

	local actionProbabilityTensorArray = {}

	local rewardValueArray = {}

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)

		local actionTensor = Model{previousFeatureTensor}
		
		local actionProbabilityTensor = calculateCategoricalProbability(actionTensor)
		
		local logActionProbabilityTensor = AutomaticDifferentiationTensor.logarithm{actionProbabilityTensor}

		table.insert(actionProbabilityTensorArray, logActionProbabilityTensor)

		table.insert(rewardValueArray, rewardValue)

	end
	
	local diagonalGaussianUpdateFunction = function(previousFeatureTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local actionMeanTensor, actionStandardDeviationTensor = Model{previousFeatureTensor}
		
		local actionProbabilityTensor = calculateDiagonalGaussianProbability(actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor)
		
		table.insert(actionProbabilityTensorArray, actionProbabilityTensor)

		table.insert(rewardValueArray, rewardValue)
		
	end
	
	local episodeUpdateFunction = function(terminalStateValue)

		local rewardToGoArray = calculateRewardToGo(rewardValueArray, discountFactor)

		for h, actionProbabilityTensor in ipairs(actionProbabilityTensorArray) do
			
			local lossTensor = actionProbabilityTensor * rewardToGoArray[h]

			lossTensor:differentiate()
			
			lossTensor:destroy{true}
			
		end
		
		WeightContainer:gradientAscent()

		table.clear(actionProbabilityTensorArray)

		table.clear(rewardValueArray)

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction, diagonalGaussianUpdateFunction, episodeUpdateFunction}

end

function ReinforcementLearningModels.ActorCritic(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local ActorModel = parameterDictionary.ActorModel or parameterDictionary[1]

	local ActorWeightContainer = parameterDictionary.ActorWeightContainer or parameterDictionary[2]
	
	local CriticModel = parameterDictionary.CriticModel or parameterDictionary[3]

	local CriticWeightContainer = parameterDictionary.CriticWeightContainer or parameterDictionary[4]

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[5] or defaultDiscountFactor

	local actionProbabilityTensorArray = {}

	local rewardValueArray = {}
	
	local criticValueArray = {}

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)

		local actionTensor = ActorModel{previousFeatureTensor}
		
		local criticValue = CriticModel{previousFeatureTensor}

		local actionProbabilityTensor = calculateCategoricalProbability(actionTensor)

		local logActionProbabilityTensor = AutomaticDifferentiationTensor.logarithm{actionProbabilityTensor}

		table.insert(actionProbabilityTensorArray, logActionProbabilityTensor)

		table.insert(rewardValueArray, rewardValue)
		
		table.insert(criticValueArray, criticValue)

	end
	
	local diagonalGaussianUpdateFunction = function(previousFeatureTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local actionMeanTensor, actionStandardDeviationTensor = ActorModel{previousFeatureTensor}

		local criticValue = CriticModel{previousFeatureTensor}

		local actionProbabilityTensor = calculateDiagonalGaussianProbability(actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor)

		table.insert(actionProbabilityTensorArray, actionProbabilityTensor)

		table.insert(rewardValueArray, rewardValue)

		table.insert(criticValueArray, criticValue)

	end

	local episodeUpdateFunction = function(terminalStateValue)

		local rewardToGoArray = calculateRewardToGo(rewardValueArray, discountFactor)

		for h, actionProbabilityTensor in ipairs(actionProbabilityTensorArray) do
			
			local criticCost = CostFunctions.FastMeanSquaredError{rewardToGoArray[h], criticValueArray[h]}
			
			local actorLossTensor = actionProbabilityTensor * criticCost
			
			criticCost:differentiate()

			actorLossTensor:differentiate()
			
			criticCost:destroy{true}
			
			actorLossTensor:destroy{true}

		end

		ActorWeightContainer:gradientAscent()
		
		CriticWeightContainer:gradientDescent()

		table.clear(actionProbabilityTensorArray)

		table.clear(rewardValueArray)
		
		table.clear(criticValueArray)

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction, diagonalGaussianUpdateFunction, episodeUpdateFunction}

end

function ReinforcementLearningModels.AdvantageActorCritic(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local ActorModel = parameterDictionary.ActorModel or parameterDictionary[1]

	local ActorWeightContainer = parameterDictionary.ActorWeightContainer or parameterDictionary[2]

	local CriticModel = parameterDictionary.CriticModel or parameterDictionary[3]

	local CriticWeightContainer = parameterDictionary.CriticWeightContainer or parameterDictionary[4]

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[5] or defaultDiscountFactor

	local actionProbabilityTensorArray = {}

	local advantageValueArray = {}

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)

		local actionTensor = ActorModel{previousFeatureTensor}

		local previousCriticValue = CriticModel{previousFeatureTensor}
		
		local currentCriticValue = CriticModel{currentFeatureTensor}

		local actionProbabilityTensor = calculateCategoricalProbability(actionTensor)
		
		local advantageValue = rewardValue + (discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		local logActionProbabilityTensor = AutomaticDifferentiationTensor.logarithm{actionProbabilityTensor}

		table.insert(actionProbabilityTensorArray, logActionProbabilityTensor)

		table.insert(advantageValueArray, advantageValue)

	end
	
	local diagonalGaussianUpdateFunction = function(previousFeatureTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

		local actionMeanTensor, actionStandardDeviationTensor = ActorModel{previousFeatureTensor}

		local previousCriticValue = CriticModel{previousFeatureTensor}

		local currentCriticValue = CriticModel{currentFeatureTensor}

		local actionProbabilityTensor = calculateDiagonalGaussianProbability(actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor)
		
		local advantageValue = rewardValue + (discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(actionProbabilityTensorArray, actionProbabilityTensor)

		table.insert(advantageValueArray, advantageValue)

	end

	local episodeUpdateFunction = function(terminalStateValue)

		for h, actionProbabilityTensor in ipairs(actionProbabilityTensorArray) do
			
			local advantageValue = advantageValueArray[h]
			
			local actorLossTensor = actionProbabilityTensor * advantageValue
			
			advantageValue:differentiate()

			actorLossTensor:differentiate()
			
			advantageValue:destroy{true}

			actorLossTensor:destroy{true}

		end

		ActorWeightContainer:gradientAscent()

		CriticWeightContainer:gradientDescent()

		table.clear(actionProbabilityTensorArray)

		table.clear(advantageValueArray)

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction, diagonalGaussianUpdateFunction, episodeUpdateFunction}

end

function ReinforcementLearningModels.ProximalPolicyOptimization(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local ActorModel = parameterDictionary.ActorModel or parameterDictionary[1]

	local ActorWeightContainer = parameterDictionary.ActorWeightContainer or parameterDictionary[2]

	local CriticModel = parameterDictionary.CriticModel or parameterDictionary[3]

	local CriticWeightContainer = parameterDictionary.CriticWeightContainer or parameterDictionary[4]

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[5] or defaultDiscountFactor
	
	local currentActorWeightTensorArray = parameterDictionary.currentActorWeightTensorArray or parameterDictionary[6] or ActorWeightContainer:getWeightTensorArray{true} -- So that the changes are reflected to the weight tensors that are put into the WeightContainer.

	local oldActorWeightTensorArray = parameterDictionary.oldActorWeightTensorArray or parameterDictionary[7] or ActorWeightContainer:getWeightTensorArray{} -- To ensure that a copy is made to avoid gradient contribution to the current actor weight tensors.

	local ratioActionProbabilityTensorArray = {}
	
	local advantageValueArray = {}
	
	local criticValueArray = {}
	
	local rewardValueArray = {}
	
	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)
		
		ActorWeightContainer:setWeightTensorArray{currentActorWeightTensorArray, true}
		
		local currentPolicyActionTensor = ActorModel{previousFeatureTensor}
		
		local currentPolicyActionProbabilityTensor = calculateCategoricalProbability(currentPolicyActionTensor)
		
		ActorWeightContainer:setWeightTensorArray{oldActorWeightTensorArray, true}
		
		local oldPolicyActionTensor = ActorModel{previousFeatureTensor}
		
		local oldPolicyActionProbabilityTensor = calculateCategoricalProbability(oldPolicyActionTensor)
		
		local ratioActionProbabilityTensor = currentPolicyActionProbabilityTensor / oldPolicyActionProbabilityTensor

		local previousCriticValue = CriticModel{previousFeatureTensor}

		local currentCriticValue = CriticModel{currentFeatureTensor}

		local advantageValue = rewardValue + (discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(ratioActionProbabilityTensorArray, ratioActionProbabilityTensor)

		table.insert(advantageValueArray, advantageValue)
		
		table.insert(criticValueArray, previousCriticValue)
		
		table.insert(rewardValueArray, rewardValue)

	end
	
	local diagonalGaussianUpdateFunction = function(previousFeatureTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

		ActorWeightContainer:setWeightTensorArray{currentActorWeightTensorArray, true}

		local currentPolicyActionMeanTensor, currentPolicyActionStandardDeviationTensor = ActorModel{previousFeatureTensor}

		local currentPolicyActionProbabilityTensor = calculateDiagonalGaussianProbability(currentPolicyActionMeanTensor, currentPolicyActionStandardDeviationTensor, actionNoiseTensor)

		ActorWeightContainer:setWeightTensorArray{oldActorWeightTensorArray, true}

		local oldPolicyActionMeanTensor, oldPolicyActionStandardDeviationTensor = ActorModel{previousFeatureTensor}

		local oldPolicyActionProbabilityTensor = calculateDiagonalGaussianProbability(oldPolicyActionMeanTensor, oldPolicyActionStandardDeviationTensor, actionNoiseTensor)

		local ratioActionProbabilityTensor = currentPolicyActionProbabilityTensor / oldPolicyActionProbabilityTensor

		local previousCriticValue = CriticModel{previousFeatureTensor}

		local currentCriticValue = CriticModel{currentFeatureTensor}

		local advantageValue = rewardValue + (discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(ratioActionProbabilityTensorArray, ratioActionProbabilityTensor)

		table.insert(advantageValueArray, advantageValue)

		table.insert(criticValueArray, previousCriticValue)

		table.insert(rewardValueArray, rewardValue)

	end

	local episodeUpdateFunction = function(terminalStateValue)
		
		local rewardToGoArray = calculateRewardToGo(rewardValueArray, discountFactor)

		for h, ratioActionProbabilityTensor in ipairs(ratioActionProbabilityTensorArray) do
			
			local criticCost = CostFunctions.FastMeanSquaredError{criticValueArray[h], rewardValueArray[h]}

			local actorLossTensor = ratioActionProbabilityTensor * advantageValueArray[h]

			criticCost:differentiate()

			actorLossTensor:differentiate()

			criticCost:destroy{true}

			actorLossTensor:destroy{true}

		end

		ActorWeightContainer:gradientAscent()

		CriticWeightContainer:gradientDescent()

		table.clear(ratioActionProbabilityTensorArray)

		table.clear(advantageValueArray)
		
		table.clear(criticValueArray)
		
		table.clear(rewardValueArray)

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction, diagonalGaussianUpdateFunction, episodeUpdateFunction}

end

function ReinforcementLearningModels.ProximalPolicyOptimizationClip(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local ActorModel = parameterDictionary.ActorModel or parameterDictionary[1]

	local ActorWeightContainer = parameterDictionary.ActorWeightContainer or parameterDictionary[2]

	local CriticModel = parameterDictionary.CriticModel or parameterDictionary[3]

	local CriticWeightContainer = parameterDictionary.CriticWeightContainer or parameterDictionary[4]
	
	local clipRatio = parameterDictionary.clipRatio or parameterDictionary[5] or defaultClipRatio

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[6] or defaultDiscountFactor

	local currentActorWeightTensorArray = parameterDictionary.currentActorWeightTensorArray or parameterDictionary[7] or ActorWeightContainer:getWeightTensorArray{true} -- So that the changes are reflected to the weight tensors that are put into the WeightContainer.

	local oldActorWeightTensorArray = parameterDictionary.oldActorWeightTensorArray or parameterDictionary[8] or ActorWeightContainer:getWeightTensorArray{} -- To ensure that a copy is made to avoid gradient contribution to the current actor weight tensors.

	local ratioActionProbabilityTensorArray = {}

	local advantageValueArray = {}

	local criticValueArray = {}

	local rewardValueArray = {}
	
	local lowerClipRatioValue = 1 - clipRatio

	local upperClipRatioValue = 1 + clipRatio

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)

		ActorWeightContainer:setWeightTensorArray{currentActorWeightTensorArray, true}

		local currentPolicyActionTensor = ActorModel{previousFeatureTensor}

		local currentPolicyActionProbabilityTensor = calculateCategoricalProbability(currentPolicyActionTensor)

		ActorWeightContainer:setWeightTensorArray{oldActorWeightTensorArray, true}

		local oldPolicyActionTensor = ActorModel{previousFeatureTensor}

		local oldPolicyActionProbabilityTensor = calculateCategoricalProbability(oldPolicyActionTensor)

		local ratioActionProbabilityTensor = currentPolicyActionProbabilityTensor / oldPolicyActionProbabilityTensor

		local previousCriticValue = CriticModel{previousFeatureTensor}

		local currentCriticValue = CriticModel{currentFeatureTensor}

		local advantageValue = rewardValue + (discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(ratioActionProbabilityTensorArray, ratioActionProbabilityTensor)

		table.insert(advantageValueArray, advantageValue)

		table.insert(criticValueArray, previousCriticValue)

		table.insert(rewardValueArray, rewardValue)

	end
	
	local diagonalGaussianUpdateFunction = function(previousFeatureTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

		ActorWeightContainer:setWeightTensorArray{currentActorWeightTensorArray, true}

		local currentPolicyActionMeanTensor, currentPolicyActionStandardDeviationTensor = ActorModel{previousFeatureTensor}

		local currentPolicyActionProbabilityTensor = calculateDiagonalGaussianProbability(currentPolicyActionMeanTensor, currentPolicyActionStandardDeviationTensor, actionNoiseTensor)

		ActorWeightContainer:setWeightTensorArray{oldActorWeightTensorArray, true}

		local oldPolicyActionMeanTensor, oldPolicyActionStandardDeviationTensor = ActorModel{previousFeatureTensor}

		local oldPolicyActionProbabilityTensor = calculateDiagonalGaussianProbability(oldPolicyActionMeanTensor, oldPolicyActionStandardDeviationTensor, actionNoiseTensor)

		local ratioActionProbabilityTensor = currentPolicyActionProbabilityTensor / oldPolicyActionProbabilityTensor

		local previousCriticValue = CriticModel{previousFeatureTensor}

		local currentCriticValue = CriticModel{currentFeatureTensor}

		local advantageValue = rewardValue + (discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(ratioActionProbabilityTensorArray, ratioActionProbabilityTensor)

		table.insert(advantageValueArray, advantageValue)

		table.insert(criticValueArray, previousCriticValue)

		table.insert(rewardValueArray, rewardValue)

	end

	local episodeUpdateFunction = function(terminalStateValue)

		local rewardToGoArray = calculateRewardToGo(rewardValueArray, discountFactor)

		for h, ratioActionProbabilityTensor in ipairs(ratioActionProbabilityTensorArray) do
			
			local criticCost = CostFunctions.FastMeanSquaredError{criticValueArray[h], rewardValueArray[h]}
			
			local clippedRatioActionProbabilityTensor = AutomaticDifferentiationTensor.clamp{ratioActionProbabilityTensor, lowerClipRatioValue, upperClipRatioValue}
			
			local actorLossTensorPart1 = ratioActionProbabilityTensor * advantageValueArray[h]
			
			local actorLossTensorPart2 = clippedRatioActionProbabilityTensor * advantageValueArray[h]
			
			local actorLossTensor = AutomaticDifferentiationTensor.minimum{actorLossTensorPart1, actorLossTensorPart2}

			criticCost:differentiate()

			actorLossTensor:differentiate()

			criticCost:destroy{true}

			actorLossTensor:destroy{true}

		end

		ActorWeightContainer:gradientAscent()

		CriticWeightContainer:gradientDescent()

		table.clear(ratioActionProbabilityTensorArray)

		table.clear(advantageValueArray)

		table.clear(criticValueArray)

		table.clear(rewardValueArray)

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction, diagonalGaussianUpdateFunction, episodeUpdateFunction}

end

function ReinforcementLearningModels.SoftActorCritic(parameterDictionary)

	parameterDictionary = parameterDictionary or {}
	
	local ActorModel = parameterDictionary.ActorModel or parameterDictionary[1]

	local ActorWeightContainer = parameterDictionary.ActorWeightContainer or parameterDictionary[2]

	local CriticModel = parameterDictionary.CriticModel or parameterDictionary[3]

	local CriticWeightContainer = parameterDictionary.CriticWeightContainer or parameterDictionary[4]
	
	local alpha = parameterDictionary.alpha or parameterDictionary[5] or defaultAlpha

	local averagingRate = parameterDictionary.averagingRate or parameterDictionary[6] or defaultAveragingRate

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[7]  or defaultDiscountFactor

	local CriticWeightTensorArrayArray = parameterDictionary.CriticWeightTensorArrayArray or parameterDictionary[8] or {}

	CriticWeightTensorArrayArray[1] = CriticWeightTensorArrayArray[1] or CriticWeightContainer:getWeightTensorArray{true} -- So that the changes are reflected to the weight tensors that are put into the WeightContainer.

	CriticWeightTensorArrayArray[2] = CriticWeightTensorArrayArray[2] or CriticWeightContainer:getWeightTensorArray{} -- To ensure that a copy is made to avoid gradient contribution to the current actor weight tensors.
	
	local function update(previousFeatureTensor, previousLogActionProbabilityTensor, currentLogActionProbabilityTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)

		local PreviousCriticWeightTensorArrayArray = {}

		local previousLogActionProbabilityValue

		if (actionIndex) then

			previousLogActionProbabilityValue = previousLogActionProbabilityTensor[1][actionIndex]

		else

			previousLogActionProbabilityValue = previousLogActionProbabilityTensor:sum()

		end

		local currentCriticValueArray = {}

		for i = 1, 2, 1 do 

			CriticWeightContainer:setWeightTensorArray{CriticWeightTensorArrayArray[i], true}

			currentCriticValueArray[i] = CriticModel{currentFeatureTensor}[1][1] 

			local CriticWeightTensorArray = CriticWeightContainer:getWeightTensorArray{true}

			PreviousCriticWeightTensorArrayArray[i] = CriticWeightTensorArray

		end

		local minimumCurrentCriticValue = AutomaticDifferentiationTensor.minimum(currentCriticValueArray)

		local yValuePart1 = (1 - terminalStateValue) * (minimumCurrentCriticValue - (alpha * previousLogActionProbabilityValue))

		local yValue = rewardValue + (discountFactor * yValuePart1)

		local previousCriticValueArray = {}

		for i = 1, 2, 1 do 

			CriticWeightContainer:setWeightTensorArray{PreviousCriticWeightTensorArrayArray[i], true}

			local previousCriticValue = CriticModel{previousFeatureTensor}

			previousCriticValueArray[i] = previousCriticValue
			
			local criticCost = CostFunctions.FastMeanSquaredError{previousCriticValue, yValue}
			
			criticCost:differentiate()

			CriticWeightContainer:gradientAscent()

			local TargetWeightTensorArray = CriticWeightContainer:getWeightTensorArray{true}

			CriticWeightTensorArrayArray[i] = rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PreviousCriticWeightTensorArrayArray[i])

		end

		local minimumCurrentCriticValue = AutomaticDifferentiationTensor.minimum(previousCriticValueArray)

		local actorCost = alpha * previousLogActionProbabilityTensor
		
		actorCost = CostFunctions.MeanSquaredError{minimumCurrentCriticValue, actorCost}
		
		actorCost:differentiate()

		ActorWeightContainer:gradientAscent()
		
		actorCost:destroy{true}

	end

	local categoricalUpdateFunction = function(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)

		local previousActionTensor = ActorModel{previousFeatureTensor}

		local currentActionTensor = ActorModel{currentFeatureTensor, true}

		local previousActionProbabilityTensor = calculateCategoricalProbability(previousActionTensor)

		local currentActionProbabilityTensor = calculateCategoricalProbability(currentActionTensor)

		local previousLogActionProbabilityTensor = AutomaticDifferentiationTensor.logarithm{previousActionProbabilityTensor}

		local currentLogActionProbabilityTensor = AutomaticDifferentiationTensor.logarithm{currentActionProbabilityTensor}
		
		update(previousFeatureTensor, previousLogActionProbabilityTensor, currentLogActionProbabilityTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)

	end
	
	local diagonalGaussianUpdate = function(previousFeatureTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

		local previousActionMeanTensor, previousStandardDeviationTensor = ActorModel{previousFeatureTensor}

		local currentActionMeanTensor, currentStandardDeviationTensor = ActorModel{currentFeatureTensor}

		local previousLogActionProbabilityTensor = calculateDiagonalGaussianProbability(previousActionMeanTensor, previousStandardDeviationTensor, actionNoiseTensor)

		local currentLogActionProbabilityTensor = calculateDiagonalGaussianProbability(currentActionMeanTensor, currentStandardDeviationTensor, actionNoiseTensor)

		update(previousFeatureTensor, previousLogActionProbabilityTensor, currentLogActionProbabilityTensor, nil, rewardValue, currentFeatureTensor, terminalStateValue)

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction, diagonalGaussianUpdate}

end

function ReinforcementLearningModels.DeepDeterministicPolicyGradient(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local ActorModel = parameterDictionary.ActorModel or parameterDictionary[1]

	local ActorWeightContainer = parameterDictionary.ActorWeightContainer or parameterDictionary[2]

	local CriticModel = parameterDictionary.CriticModel or parameterDictionary[3]

	local CriticWeightContainer = parameterDictionary.CriticWeightContainer or parameterDictionary[4]

	local averagingRate = parameterDictionary.averagingRate or parameterDictionary[5] or defaultAveragingRate

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[6] or defaultDiscountFactor

	local diagonalGaussianUpdate = function(previousFeatureTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local ActorWeightTensorArray = ActorWeightContainer:getWeightTensorArray{}
		
		local CriticWeightTensorArray = CriticWeightContainer:getWeightTensorArray{}
		
		local previousActionMeanTensor, previousActionStandardDeviationTensor = ActorModel{previousFeatureTensor}

		local currentActionMeanTensor, currentActionStandardDeviationTensor = ActorModel{currentFeatureTensor}

		local targetCriticActionMeanInputTensor = AutomaticDifferentiationTensor.concatenate(currentFeatureTensor, currentActionMeanTensor, 2)

		local targetQValue = CriticModel{targetCriticActionMeanInputTensor}[1][1]

		local yValue = rewardValue + (discountFactor * (1 - terminalStateValue) * targetQValue)

		local actionTensor = (previousActionStandardDeviationTensor * actionNoiseTensor) + previousActionMeanTensor

		local previousCriticActionInputTensor = AutomaticDifferentiationTensor.concatenate(previousFeatureTensor, actionTensor, 2)

		local currentQValue = CriticModel{previousCriticActionInputTensor}[1][1]

		local negatedtemporalDifferenceError = currentQValue - yValue
		
		negatedtemporalDifferenceError:differentiate()

		ActorWeightContainer:gradientAscent()

		local previousCriticActionMeanInputTensor = AutomaticDifferentiationTensor.concatenate(previousFeatureTensor, previousActionMeanTensor, 2)
		
		previousCriticActionMeanInputTensor:differentiate()

		CriticWeightContainer:gradientDescent()
		
		negatedtemporalDifferenceError:destroy{}
		
		previousCriticActionMeanInputTensor:destroy{}

		local TargetActorWeightTensorArray = ActorWeightContainer:getWeightTensorArray{true}

		local TargetCriticWeightTensorArray = CriticWeightContainer:getWeightTensorArray{true}

		TargetActorWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetActorWeightTensorArray, ActorWeightTensorArray)

		TargetCriticWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetCriticWeightTensorArray, CriticWeightTensorArray)

		ActorWeightContainer:setWeightTensorArray{TargetActorWeightTensorArray, true}

		CriticWeightContainer:setWeightTensorArray{TargetCriticWeightTensorArray, true}

	end

	return ReinforcementLearningModels.new{nil, diagonalGaussianUpdate}

end

function ReinforcementLearningModels.TwinDelayedDeepDeterministicPolicyGradient(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local ActorModel = parameterDictionary.ActorModel or parameterDictionary[1]

	local ActorWeightContainer = parameterDictionary.ActorWeightContainer or parameterDictionary[2]

	local CriticModel = parameterDictionary.CriticModel or parameterDictionary[3]

	local CriticWeightContainer = parameterDictionary.CriticWeightContainer or parameterDictionary[4]

	local averagingRate = parameterDictionary.averagingRate or parameterDictionary[5] or defaultAveragingRate
	
	local noiseClippingFactor = parameterDictionary.noiseClippingFactor or parameterDictionary[6] or defaultNoiseClippingFactor

	local policyDelayAmount = parameterDictionary.policyDelayAmount or parameterDictionary[7] or defaultPolicyDelayAmount

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[8] or defaultDiscountFactor
	
	local CriticWeightTensorArrayArray = parameterDictionary.CriticWeightTensorArrayArray or {}
	
	local TargetCriticWeightTensorArrayArray = {}

	local currentNumberOfUpdate = 0

	local diagonalGaussianUpdate = function(previousFeatureTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

		local noiseClipFunction = function(value) return math.clamp(value, -noiseClippingFactor, noiseClippingFactor) end

		local clippedCurrentActionNoiseTensor = AqwamTensorLibrary:applyFunction(noiseClipFunction, actionNoiseTensor)
		
		local previousActionMeanTensor, previousActionStandardDeviationTensor = ActorModel{previousFeatureTensor}

		local previousActionTensor = (previousActionStandardDeviationTensor * actionNoiseTensor) + previousActionMeanTensor

		local lowestActionValue = previousActionTensor:findMinimumValue()

		local highestActionValue = previousActionTensor:findMaximumValue()

		local currentActionMeanTensor, currentActionStandardDeviationTensor = ActorModel{currentFeatureTensor}

		local ActorWeightTensorArray = ActorModel:getWeightTensorArray{true}

		local targetActionTensorPart1 = currentActionMeanTensor + clippedCurrentActionNoiseTensor

		local actionClipFunction = function(value)

			if (lowestActionValue ~= lowestActionValue) or (highestActionValue ~= highestActionValue) then

				error("Received nan values.")

			elseif (lowestActionValue < highestActionValue) then

				return math.clamp(value, lowestActionValue, highestActionValue) 

			elseif (lowestActionValue > highestActionValue) then

				return math.clamp(value, highestActionValue, lowestActionValue)

			else

				return lowestActionValue

			end

		end

		local targetActionTensor = AqwamTensorLibrary:applyFunction(actionClipFunction, targetActionTensorPart1)

		local targetCriticActionInputTensor = AutomaticDifferentiationTensor.concatenate(currentFeatureTensor, targetActionTensor, 2)

		local currentCriticValueArray = {}

		for i = 1, 2, 1 do 

			CriticWeightContainer:setWeightTensorArray{TargetCriticWeightTensorArrayArray[i]}

			currentCriticValueArray[i] = CriticModel{targetCriticActionInputTensor}[1][1] 

			local CriticWeightTensorArray = CriticWeightContainer:getWeightTensorArray{true}

			TargetCriticWeightTensorArrayArray[i] = CriticWeightTensorArray

		end

		local minimumCurrentCriticValue = AutomaticDifferentiationTensor.minimum(currentCriticValueArray)

		local yValuePart1 = discountFactor * (1 - terminalStateValue) * minimumCurrentCriticValue

		local yValue = rewardValue + yValuePart1

		local previousCriticActionMeanInputTensor = AutomaticDifferentiationTensor.concatenate(previousFeatureTensor, previousActionMeanTensor, 2)

		for i = 1, 2, 1 do 

			CriticWeightContainer:setWeightTensorArray{CriticWeightTensorArrayArray[i], true}

			local previousCriticValue = CriticModel{previousCriticActionMeanInputTensor}[1][1] 

			local criticCost = previousCriticValue - yValue
			
			criticCost:differentiate()

			CriticWeightContainer:gradientDescent()

			CriticWeightTensorArrayArray[i] = CriticWeightContainer:getWeightTensorArray{true}

		end

		currentNumberOfUpdate = currentNumberOfUpdate + 1
		
		if ((currentNumberOfUpdate % policyDelayAmount) == 0) then

			local actionTensor = (previousActionStandardDeviationTensor * actionNoiseTensor) + previousActionMeanTensor

			local previousCriticActionInputTensor = AutomaticDifferentiationTensor.concatenate(previousFeatureTensor, actionTensor, 2)

			CriticWeightContainer:setWeightTensorArray{CriticWeightTensorArrayArray[1], true}

			local currentQValue = CriticModel{previousCriticActionInputTensor}[1][1]

			currentQValue:differentiate()

			ActorWeightContainer:gradientAscent()

			for i = 1, 2, 1 do TargetCriticWeightTensorArrayArray[i] = rateAverageWeightTensorArray(averagingRate, TargetCriticWeightTensorArrayArray[i], CriticWeightTensorArrayArray[i]) end

			local TargetActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

			TargetActorWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetActorWeightTensorArray, ActorWeightTensorArray)

			ActorWeightContainer:setWeightTensorArray{TargetActorWeightTensorArray, true}

		end

	end

	return ReinforcementLearningModels.new{nil, diagonalGaussianUpdate}

end

function ReinforcementLearningModels:categoricalUpdate(parameterDictionary)
	
	local categoricalUpdateFunction = self.categoricalUpdateFunction
	
	if (not categoricalUpdateFunction) then
		
		error("The reinforcement learning model does not support categorical updates.")
		
	end
	
	parameterDictionary = parameterDictionary or {}
	
	local previousFeatureTensor = parameterDictionary.previousFeatureTensor or parameterDictionary[1]

	local actionIndex = parameterDictionary.actionIndex or parameterDictionary[2]

	local rewardValue = parameterDictionary.rewardValue or parameterDictionary[3]

	local currentFeatureTensor = parameterDictionary.currentFeatureTensor or parameterDictionary[4]
	
	local terminalStateValue = parameterDictionary.terminalStateValue or parameterDictionary[5]
	
	return categoricalUpdateFunction(previousFeatureTensor, actionIndex, rewardValue, currentFeatureTensor, terminalStateValue)
	
end

function ReinforcementLearningModels:diagonalGaussianUpdate(parameterDictionary)

	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction

	if (not diagonalGaussianUpdateFunction) then

		error("The reinforcement learning model does not support diagonal gaussian updates.")

	end

	parameterDictionary = parameterDictionary or {}

	local previousFeatureTensor = parameterDictionary.previousFeatureTensor or parameterDictionary[1]
	
	local actionNoiseTensor = parameterDictionary.actionNoiseTensor or parameterDictionary[2]

	local rewardValue = parameterDictionary.rewardValue or parameterDictionary[3]

	local currentFeatureTensor = parameterDictionary.currentFeatureTensor or parameterDictionary[4]

	local terminalStateValue = parameterDictionary.terminalStateValue or parameterDictionary[5]

	return diagonalGaussianUpdateFunction(previousFeatureTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

end

function ReinforcementLearningModels:episodeUpdate(parameterDictionary)
	
	local episodeUpdateFunction = self.episodeUpdateFunction
	
	if (not episodeUpdateFunction) then return end
	
	parameterDictionary = parameterDictionary or {}
	
	local terminalStateValue = parameterDictionary.terminalStateValue or parameterDictionary[1] or 1
	
	return episodeUpdateFunction(terminalStateValue)
	
end

return ReinforcementLearningModels
