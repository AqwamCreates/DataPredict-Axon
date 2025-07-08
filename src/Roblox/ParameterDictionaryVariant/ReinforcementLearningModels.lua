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

local defaultClipRatio = 0.3

local defaultDiscountFactor = 0.95

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
		
		local cost = CostFunctions.FastMeanSquaredError{targetValue, previousQValueTensor}
		
		cost:differentiate()

		WeightContainer:gradientAscent()
		
		cost:destroy{true}
		
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
		
		costTensor:destroy{true}

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction}

end

function ReinforcementLearningModels.DeepExpectedStateActionRewardStateAction(parameterDictionary)

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

	return ReinforcementLearningModels.new{categoricalUpdateFunction, nil, episodeUpdateFunction}

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

	local episodeUpdateFunction = function(terminalStateValue)

		local rewardToGoArray = calculateRewardToGo(rewardValueArray, discountFactor)

		for h, actionProbabilityTensor in ipairs(actionProbabilityTensorArray) do
			
			local criticLoss = rewardToGoArray[h] - criticValueArray[h]
			
			local actorLossTensor = actionProbabilityTensor * criticLoss
			
			criticLoss:differentiate()

			actorLossTensor:differentiate()
			
			criticLoss:destroy{true}
			
			actorLossTensor:destroy{true}

		end

		ActorWeightContainer:gradientAscent()
		
		CriticWeightContainer:gradientAscent()

		table.clear(actionProbabilityTensorArray)

		table.clear(rewardValueArray)
		
		table.clear(criticValueArray)

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction, nil, episodeUpdateFunction}

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

		CriticWeightContainer:gradientAscent()

		table.clear(actionProbabilityTensorArray)

		table.clear(advantageValueArray)

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction, nil, episodeUpdateFunction}

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

	local episodeUpdateFunction = function(terminalStateValue)
		
		local rewardToGoArray = calculateRewardToGo(rewardValueArray, discountFactor)

		for h, ratioActionProbabilityTensor in ipairs(ratioActionProbabilityTensorArray) do
			
			local criticLoss = CostFunctions.FastMeanSquaredError{criticValueArray[h], rewardValueArray[h]}

			local actorLossTensor = ratioActionProbabilityTensor * advantageValueArray[h]

			criticLoss:differentiate()

			actorLossTensor:differentiate()

			criticLoss:destroy{true}

			actorLossTensor:destroy{true}

		end

		ActorWeightContainer:gradientAscent()

		CriticWeightContainer:gradientDescent()

		table.clear(ratioActionProbabilityTensorArray)

		table.clear(advantageValueArray)
		
		table.clear(criticValueArray)
		
		table.clear(rewardValueArray)

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction, nil, episodeUpdateFunction}

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

	local episodeUpdateFunction = function(terminalStateValue)

		local rewardToGoArray = calculateRewardToGo(rewardValueArray, discountFactor)

		for h, ratioActionProbabilityTensor in ipairs(ratioActionProbabilityTensorArray) do
			
			local criticLoss = CostFunctions.FastMeanSquaredError{criticValueArray[h], rewardValueArray[h]}
			
			local clippedRatioActionProbabilityTensor = AutomaticDifferentiationTensor.clamp{ratioActionProbabilityTensor, lowerClipRatioValue, upperClipRatioValue}
			
			local actorLossTensorPart1 = ratioActionProbabilityTensor * advantageValueArray[h]
			
			local actorLossTensorPart2 = clippedRatioActionProbabilityTensor * advantageValueArray[h]
			
			local actorLossTensor = AutomaticDifferentiationTensor.minimum{actorLossTensorPart1, actorLossTensorPart2}

			criticLoss:differentiate()

			actorLossTensor:differentiate()

			criticLoss:destroy{true}

			actorLossTensor:destroy{true}

		end

		ActorWeightContainer:gradientAscent()

		CriticWeightContainer:gradientDescent()

		table.clear(ratioActionProbabilityTensorArray)

		table.clear(advantageValueArray)

		table.clear(criticValueArray)

		table.clear(rewardValueArray)

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction, nil, episodeUpdateFunction}

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

	local actionMeanTensor = parameterDictionary.actionMeanTensor or parameterDictionary[2]
	
	local actionStandardDeviationTensor = parameterDictionary.actionStandardDeviationTensor or parameterDictionary[3]
	
	local actionNoiseTensor = parameterDictionary.actionNoiseTensor or parameterDictionary[4]

	local rewardValue = parameterDictionary.rewardValue or parameterDictionary[5]

	local currentFeatureTensor = parameterDictionary.currentFeatureTensor or parameterDictionary[6]

	local terminalStateValue = parameterDictionary.terminalStateValue or parameterDictionary[7]

	return diagonalGaussianUpdateFunction(previousFeatureTensor, actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

end

function ReinforcementLearningModels:episodeUpdate(parameterDictionary)
	
	local episodeUpdateFunction = self.episodeUpdateFunction
	
	if (not episodeUpdateFunction) then return end
	
	parameterDictionary = parameterDictionary or {}
	
	local terminalStateValue = parameterDictionary.terminalStateValue or parameterDictionary[1] or 1
	
	return episodeUpdateFunction(terminalStateValue)
	
end

return ReinforcementLearningModels
