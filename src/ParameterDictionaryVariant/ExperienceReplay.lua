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

local ExperienceReplay = {}

ExperienceReplay.__index = ExperienceReplay

local defaultBatchSize = 32

local defaultAlpha = 0.6

local defaultBeta = 0.4

local defaultAggregateFunction = "Maximum"

local defaultEpsilon = math.pow(10, -4)

local aggregrateFunctionList = {

	["Maximum"] = function (valueVector) 

		return AqwamTensorLibrary:findMaximumValue(valueVector) 

	end,

	["Minimum"] = function (valueVector) 

		return AqwamTensorLibrary:findMinimumValue(valueVector) 

	end,

	["Sum"] = function (valueVector) 

		return AqwamTensorLibrary:sum(valueVector) 

	end,

	["Average"] = function (valueVector) 

		return AqwamTensorLibrary:sum(valueVector) / #valueVector[1] 

	end,

}

local function sample(replayBufferArray, batchSize)

	local batchArray = {}

	local replayBufferArray = replayBufferArray

	local replayBufferArraySize = #replayBufferArray

	local lowestNumberOfBatchSize = math.min(batchSize, replayBufferArraySize)

	for i = 1, lowestNumberOfBatchSize, 1 do

		table.insert(batchArray, replayBufferArray[i])

	end

	return batchArray

end


function ExperienceReplay.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewExperienceReplay = {}

	setmetatable(NewExperienceReplay, ExperienceReplay)
	
	NewExperienceReplay.RunFunction = parameterDictionary.RunFunction or parameterDictionary[1]

	NewExperienceReplay.numberOfRunsToUpdate = parameterDictionary.numberOfRunsToUpdate or parameterDictionary[3] or 1

	NewExperienceReplay.maximumBufferSize = parameterDictionary.maximumBufferSize or parameterDictionary[4] or 100

	NewExperienceReplay.numberOfRuns = parameterDictionary.numberOfRuns or parameterDictionary[2] or 0

	return NewExperienceReplay

end

function ExperienceReplay.UniformExperienceReplay(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local batchSize = parameterDictionary.batchSize or parameterDictionary[1] or defaultBatchSize

	local numberOfRunsToUpdate = parameterDictionary.numberOfRunsToUpdate or parameterDictionary[2] 

	local maximumBufferSize = parameterDictionary.maximumBufferSize or parameterDictionary[3]

	local numberOfRuns = parameterDictionary.numberOfRuns or parameterDictionary[4] 

	local replayBufferArray = parameterDictionary.replayBufferArray or parameterDictionary[5] or {}
	
	local RunFunction = function(UpdateFunction)
		
		local experienceReplayBatchArray = sample(replayBufferArray, batchSize)

		for _, experience in ipairs(experienceReplayBatchArray) do UpdateFunction(table.unpack(experience)) end
		
	end
	
	return ExperienceReplay.new({RunFunction, numberOfRunsToUpdate, maximumBufferSize, numberOfRuns})
	
end


function ExperienceReplay.NStepExperienceReplay(parameterDictionary)

	parameterDictionary = parameterDictionary or {}
	
	local nStep = parameterDictionary.nStep or parameterDictionary[1] or 3
	
	local batchSize = parameterDictionary.batchSize or parameterDictionary[2] or defaultBatchSize

	local numberOfRunsToUpdate = parameterDictionary.numberOfRunsToUpdate or parameterDictionary[3] 

	local maximumBufferSize = parameterDictionary.maximumBufferSize or parameterDictionary[4]

	local numberOfRuns = parameterDictionary.numberOfRuns or parameterDictionary[5] 

	local replayBufferArray = parameterDictionary.replayBufferArray or parameterDictionary[6] or {}

	local RunFunction = function(UpdateFunction)

		local replayBufferBatchArray = sample(replayBufferArray, batchSize)

		local replayBufferArraySize = #replayBufferArray

		local replayBufferBatchArraySize = #replayBufferBatchArray

		local nStep = math.min(nStep, replayBufferBatchArraySize)

		local finalBatchArrayIndex = (replayBufferBatchArraySize - nStep) + 1

		for i = replayBufferBatchArraySize, finalBatchArrayIndex, -1 do UpdateFunction(table.unpack(replayBufferBatchArray[i])) end

	end

	return ExperienceReplay.new({RunFunction, numberOfRunsToUpdate, maximumBufferSize, numberOfRuns})

end

function ExperienceReplay.PrioritizedExperienceReplay(parameterDictionary)

	parameterDictionary = parameterDictionary or {}
	
	local Model = parameterDictionary.Model

	local alpha = parameterDictionary.alpha

	local beta = parameterDictionary.beta

	local epsilon = parameterDictionary.epsilon

	local replayBufferArray = parameterDictionary.replayBufferArray

	local temporalDifferenceArray = parameterDictionary.temporalDifferenceErrorArray

	local priorityArray = parameterDictionary.priorityArray or {}

	local weightArray = parameterDictionary.weightArray or {}
	
	local aggregateFunctionToApply = aggregrateFunctionList[aggregateFunction]

	local addExperienceFunction = function()

		local maxPriority = 1

		for i, priority in ipairs(priorityArray) do

			if (priority <= maxPriority) then continue end

			maxPriority = priority

		end

		table.insert(priorityArray, maxPriority)

		table.insert(weightArray, 0)

		NewPrioritizedExperienceReplay:removeFirstValueFromArrayIfExceedsBufferSize(priorityArray)

		NewPrioritizedExperienceReplay:removeFirstValueFromArrayIfExceedsBufferSize(weightArray)

	end

	NewPrioritizedExperienceReplay:extendResetFunction(function()

		table.clear(priorityArray)

		table.clear(weightArray)

	end)

	local RunFunction = (function(UpdateFunction, replayBufferArray, batchSize)

		if (not Model) then error("No Model!") end

		local batchArray = {}

		local replayBufferArraySize = #replayBufferArray

		local lowestNumberOfBatchSize = math.min(batchSize, replayBufferArraySize)		

		local probabilityArray = {}

		local sumPriorityAlpha = 0

		for i, priority in ipairs(priorityArray) do

			local priorityAlpha = math.pow(priority, alpha)

			probabilityArray[i] = priorityAlpha

			sumPriorityAlpha = sumPriorityAlpha + priorityAlpha

		end

		for i, probability in ipairs(probabilityArray) do

			probabilityArray[i] = probability / sumPriorityAlpha

		end

		local sizeArray = AqwamTensorLibrary:getDimensionSizeArray(replayBufferArray[1][1])

		local inputMatrix = AqwamTensorLibrary:createTensor(sizeArray, 1)

		local sumLossMatrix

		for i = 1, lowestNumberOfBatchSize, 1 do

			local index, probability = sample(probabilityArray, sumPriorityAlpha)

			local experience = replayBufferArray[index]

			local temporalDifferenceErrorValueOrVector = temporalDifferenceArray[index]

			local importanceSamplingWeight = math.pow((lowestNumberOfBatchSize * probability), -beta) / math.max(table.unpack(weightArray), epsilon) 

			if (type(temporalDifferenceErrorValueOrVector) ~= "number") then

				temporalDifferenceErrorValueOrVector = aggregateFunctionToApply(temporalDifferenceErrorValueOrVector)

			end

			weightArray[index] = importanceSamplingWeight

			priorityArray[index] = math.abs(temporalDifferenceErrorValueOrVector)

			local outputMatrix = Model:forwardPropagate(replayBufferArray[i][1], false)

			local lossMatrix = AqwamTensorLibrary:multiply(outputMatrix, temporalDifferenceErrorValueOrVector, importanceSamplingWeight)

			if (sumLossMatrix) then

				sumLossMatrix = AqwamTensorLibrary:add(sumLossMatrix, lossMatrix)

			else

				sumLossMatrix = lossMatrix

			end

		end

		Model:forwardPropagate(inputMatrix, true)

		Model:update(sumLossMatrix, true)

	end)

	return ExperienceReplay.new({RunFunction})

end

function ExperienceReplay:reset()

	self.numberOfRuns = 0

	table.clear(self.replayBufferArray)

	table.clear(self.temporalDifferenceErrorArray)

	local ResetFunction = self.ResetFunction

	if ResetFunction then ResetFunction() end

end

function ExperienceReplay:run(updateFunction)
	
	local numberOfRuns = self.numberOfRuns + 1

	self.numberOfRuns = numberOfRuns

	if (numberOfRuns < self.numberOfRunsToUpdate) then return nil end

	self.numberOfRuns = 0

	self.RunFunction(updateFunction)

end

function ExperienceReplay:removeFirstValueFromArrayIfExceedsBufferSize(targetArray)

	if (#targetArray > self.maxBufferSize) then table.remove(targetArray, 1) end

end

function ExperienceReplay:extendAddExperienceFunction(AddExperienceFunction)

	self.AddExperienceFunction = AddExperienceFunction

end

function ExperienceReplay:addExperience(...)

	local experience = {...}
	
	local replayBufferArray = self.replayBufferArray

	table.insert(replayBufferArray, experience)

	local addExperienceFunction = self.addExperienceFunction

	if (addExperienceFunction) then addExperienceFunction(...) end

	self:removeFirstValueFromArrayIfExceedsBufferSize(replayBufferArray)

end

function ExperienceReplay:addTemporalDifferenceError(temporalDifferenceErrorVectorOrValue)

	if (not self.isTemporalDifferenceErrorRequired) then return end
	
	local temporalDifferenceErrorArray = self.temporalDifferenceErrorArray

	table.insert(temporalDifferenceErrorArray, temporalDifferenceErrorVectorOrValue)

	local AddTemporalDifferenceErrorFunction = self.AddTemporalDifferenceErrorFunction

	if (AddTemporalDifferenceErrorFunction) then AddTemporalDifferenceErrorFunction(temporalDifferenceErrorVectorOrValue) end

	self:removeFirstValueFromArrayIfExceedsBufferSize(temporalDifferenceErrorArray)

end

return ExperienceReplay
