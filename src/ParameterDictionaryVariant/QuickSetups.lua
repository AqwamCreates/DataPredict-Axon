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

local DisplayErrorFunctions = require(script.Parent.DisplayErrorFunctions)

local displayFunctionErrorDueToNonObjectCondition = DisplayErrorFunctions.displayFunctionErrorDueToNonObjectCondition

local QuickSetup = {}

QuickSetup.__index = QuickSetup

local defaultNumberOfReinforcementsPerEpisode = 500

function QuickSetup.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewQuickSetup = {}

	setmetatable(NewQuickSetup, QuickSetup)

	NewQuickSetup.reinforceFunction = parameterDictionary.reinforceFunction or parameterDictionary[1]
	
	NewQuickSetup.resetFunction = parameterDictionary.resetFunction or parameterDictionary[2]

	NewQuickSetup.isAnObject = true

	return NewQuickSetup

end

function QuickSetup.CategoricalQuickSetup(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local Model = parameterDictionary.Model or parameterDictionary[1]
	
	local numberOfReinforcementsPerEpisode = parameterDictionary.numberOfReinforcementsPerEpisode or parameterDictionary[2] or defaultNumberOfReinforcementsPerEpisode
	
	local ExperienceReplay = parameterDictionary.ExperienceReplay or parameterDictionary[3]
	
	local updateFunction = parameterDictionary.updateFunction or parameterDictionary[4]
	
	local episodeUpdateFunction = parameterDictionary.episodeUpdateFunction or parameterDictionary[5]
	
	local currentNumberOfReinforcements = 0
	
	local currentNumberOfEpisodes = 0
	
	local previousFeatureTensor
	
	local previousActionIndex

	local reinforceFunction = function(currentFeatureTensor, rewardValue, returnOriginalOutput)

		currentNumberOfReinforcements = currentNumberOfReinforcements + 1

		local currentActionTensor = Model{currentFeatureTensor}

		local isEpisodeEnd = (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode)

		local terminalStateValue = (isEpisodeEnd and 1) or 0

		local currentActionIndex

		local currentActionValue = currentActionTensor[1][currentActionIndex]

		local temporalDifferenceError

		if (previousFeatureTensor) then

			temporalDifferenceError = Model:categoricalUpdate{previousFeatureTensor, previousActionIndex, rewardValue, currentFeatureTensor, currentActionIndex, terminalStateValue}

			if (updateFunction) then updateFunction(terminalStateValue) end

		end

		if (isEpisodeEnd) then

			currentNumberOfReinforcements = 0

			currentNumberOfEpisodes = currentNumberOfEpisodes + 1

			Model:episodeUpdate(terminalStateValue)

			if (episodeUpdateFunction) then episodeUpdateFunction(terminalStateValue) end

		end

		if (ExperienceReplay) and (previousFeatureTensor) then

			ExperienceReplay:addExperience{previousFeatureTensor, previousActionIndex, rewardValue, currentFeatureTensor, currentActionIndex, terminalStateValue}

			ExperienceReplay:addTemporalDifferenceError{temporalDifferenceError}

			ExperienceReplay:run{function(storedPreviousFeatureTensor, storedPreviousActionIndex, storedRewardValue, storedCurrentFeatureTensor, storedCurrentActionIndex, storedTerminalStateValue)

				return Model:categoricalUpdate{storedPreviousFeatureTensor, storedPreviousActionIndex, storedRewardValue, storedCurrentFeatureTensor, storedCurrentActionIndex, storedTerminalStateValue}

			end}

		end

		previousFeatureTensor = currentFeatureTensor

		previousActionIndex = currentActionIndex

		if (returnOriginalOutput) then return currentActionTensor end

		return currentActionIndex, currentActionValue

	end
	
	local resetFunction = function()
		
		currentNumberOfReinforcements = 0
		
		currentNumberOfEpisodes = 0
		
		previousFeatureTensor = nil
		
		previousActionIndex = nil
		
	end

	return QuickSetup.new({reinforceFunction, resetFunction})

end

function QuickSetup.DiagonalGaussian(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local Model = parameterDictionary.Model or parameterDictionary[1]

	local numberOfReinforcementsPerEpisode = parameterDictionary.numberOfReinforcementsPerEpisode or parameterDictionary[2] or defaultNumberOfReinforcementsPerEpisode

	local ExperienceReplay = parameterDictionary.ExperienceReplay or parameterDictionary[3]
	
	local updateFunction = parameterDictionary.updateFunction or parameterDictionary[4]

	local episodeUpdateFunction = parameterDictionary.episodeUpdateFunction or parameterDictionary[5]

	local currentNumberOfReinforcements = 0

	local currentNumberOfEpisodes = 0

	local previousFeatureTensor

	local previousActionTensor

	local reinforceFunction = function(currentFeatureTensor, rewardValue, returnOriginalOutput)

		currentNumberOfReinforcements = currentNumberOfReinforcements + 1

		local currentActionTensor = Model{currentFeatureTensor}

		local isEpisodeEnd = (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode)

		local terminalStateValue = (isEpisodeEnd and 1) or 0

		local temporalDifferenceError

		if (previousFeatureTensor) then

			temporalDifferenceError = Model:diagonalGaussianUpdate{previousFeatureTensor, previousActionTensor, rewardValue, currentFeatureTensor, currentActionTensor, terminalStateValue}

			if (updateFunction) then updateFunction(terminalStateValue) end

		end

		if (isEpisodeEnd) then

			currentNumberOfReinforcements = 0

			currentNumberOfEpisodes = currentNumberOfEpisodes + 1

			Model:episodeUpdate(terminalStateValue)

			if (episodeUpdateFunction) then episodeUpdateFunction(terminalStateValue) end

		end

		if (ExperienceReplay) and (previousFeatureTensor) then

			ExperienceReplay:addExperience{previousFeatureTensor, previousActionTensor, rewardValue, currentFeatureTensor, currentActionTensor, terminalStateValue}

			ExperienceReplay:addTemporalDifferenceError{temporalDifferenceError}

			ExperienceReplay:run{function(storedPreviousFeatureTensor, storedPreviousActionIndex, storedRewardValue, storedCurrentFeatureTensor, storedCurrentActionIndex, storedTerminalStateValue)

				return Model:diagonalGaussianUpdate{storedPreviousFeatureTensor, storedPreviousActionIndex, storedRewardValue, storedCurrentFeatureTensor, storedCurrentActionIndex, storedTerminalStateValue}

			end}

		end

		previousFeatureTensor = currentFeatureTensor

		previousActionTensor = currentActionTensor

		return currentActionTensor

	end
	
	local resetFunction = function()

		currentNumberOfReinforcements = 0

		currentNumberOfEpisodes = 0

		previousFeatureTensor = nil

		previousActionTensor = nil

	end

	return QuickSetup.new({reinforceFunction, resetFunction})

end

function QuickSetup:reinforce(parameterDictionary)
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	parameterDictionary = parameterDictionary or {}
	
	local featureTensor = parameterDictionary.featureTensor or parameterDictionary[1]
	
	local rewardValue = parameterDictionary.rewardValue or parameterDictionary[2]
	
	local returnOriginalOutput = parameterDictionary.returnOriginalOutput or parameterDictionary[3]
	
	return self.reinforceFunction(featureTensor, rewardValue, returnOriginalOutput)

end

function QuickSetup:reset()

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	return self.resetFunction()

end

return QuickSetup
