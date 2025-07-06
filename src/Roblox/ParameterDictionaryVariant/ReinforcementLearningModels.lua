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
	
	local categoricalUpdateFunction = function(previousStateTensor, actionIndex, rewardValue, currentStateTensor, terminalStateValue)
		
		local previousQValueTensor = Model{previousStateTensor}
		
		local currentQValueTensor = Model{currentStateTensor}
		
		local maxQValue = currentQValueTensor:findMaximumValue()
		
		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue)
		
		local lastValue = previousQValueTensor[1][actionIndex]
		
		local loss = CostFunctions.FastMeanSquaredError{targetValue, previousQValueTensor}
		
		loss:differentiate()

		WeightContainer:gradientAscent()
		
		return loss
		
	end
	
	return ReinforcementLearningModels.new{categoricalUpdateFunction}
	
end

function ReinforcementLearningModels.DeepStateActionRewardStateAction(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local WeightContainer = parameterDictionary.WeightContainer or parameterDictionary[2]

	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[3] or defaultDiscountFactor

	local categoricalUpdateFunction = function(previousStateTensor, actionIndex, rewardValue, currentStateTensor, terminalStateValue)

		local previousQValueTensor = Model{previousStateTensor}

		local currentQValueTensor = Model{currentStateTensor}

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * currentQValueTensor)

		local lastValue = previousQValueTensor[1][actionIndex]

		local lossTensor = CostFunctions.FastMeanSquaredError{currentQValueTensor, previousQValueTensor}

		lossTensor:differentiate()

		WeightContainer:gradientAscent()
		
		return lossTensor

	end

	return ReinforcementLearningModels.new{categoricalUpdateFunction}

end

function ReinforcementLearningModels:categoricalUpdate(stateParameterDictionary)
	
	local categoricalUpdateFunction = self.categoricalUpdateFunction
	
	if (not categoricalUpdateFunction) then
		
		error("This reinforcement learning model does not support categorical updates.")
		
	end
	
	stateParameterDictionary = stateParameterDictionary or {}
	
	local previousStateTensor = stateParameterDictionary.previousStateTensor or stateParameterDictionary[1]

	local actionIndex = stateParameterDictionary.actionIndex or stateParameterDictionary[2]

	local rewardValue = stateParameterDictionary.rewardValue or stateParameterDictionary[3]

	local currentStateTensor = stateParameterDictionary.currentStateTensor or stateParameterDictionary[4]
	
	local terminalStateValue = stateParameterDictionary.terminalStateValue or stateParameterDictionary[5]
	
	return categoricalUpdateFunction(previousStateTensor, actionIndex, rewardValue, currentStateTensor, terminalStateValue)
	
end

function ReinforcementLearningModels:diagonalGaussianUpdate(stateParameterDictionary)

	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction

	if (not diagonalGaussianUpdateFunction) then

		error("This reinforcement learning model does not support diagonal gaussian updates.")

	end

	stateParameterDictionary = stateParameterDictionary or {}

	local previousStateTensor = stateParameterDictionary.previousStateTensor or stateParameterDictionary[1]

	local actionIndex = stateParameterDictionary.actionIndex or stateParameterDictionary[2]

	local rewardValue = stateParameterDictionary.rewardValue or stateParameterDictionary[3]

	local currentStateTensor = stateParameterDictionary.currentStateTensor or stateParameterDictionary[4]

	local terminalStateValue = stateParameterDictionary.terminalStateValue or stateParameterDictionary[5]

	return diagonalGaussianUpdateFunction(previousStateTensor, actionIndex, rewardValue, currentStateTensor, terminalStateValue)

end

function ReinforcementLearningModels:episodeUpdate(stateParameterDictionary)
	
	local episodeUpdateFunction = self.episodeUpdateFunction
	
	if (not episodeUpdateFunction) then return end
	
	stateParameterDictionary = stateParameterDictionary or {}
	
	local terminalStateValue = stateParameterDictionary.terminalStateValue or stateParameterDictionary[1] or 1
	
	return episodeUpdateFunction(terminalStateValue)
	
end

return ReinforcementLearningModels
