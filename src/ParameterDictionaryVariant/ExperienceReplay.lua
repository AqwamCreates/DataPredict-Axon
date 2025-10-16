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

local aggregrateFunctionList = {

	["Maximum"] = function (valueVector) return valueVector:findMaximumValue() end,

	["Minimum"] = function (valueVector) return valueVector:findMinimumValue() end,

	["Sum"] = function (valueVector) return valueVector:sum() end,

	["Average"] = function (valueVector) return valueVector:mean() end,

}

local function removeFirstValueFromArrayIfExceedsBufferSize(targetArray, maximumBufferSize)

	if (#targetArray > maximumBufferSize) then table.remove(targetArray, 1) end

end


local function sampleBuffer(replayBufferArray, batchSize)

	local batchArray = {}

	local replayBufferArray = replayBufferArray

	local replayBufferArraySize = #replayBufferArray

	local lowestNumberOfBatchSize = math.min(batchSize, replayBufferArraySize)

	for i = 1, lowestNumberOfBatchSize, 1 do

		table.insert(batchArray, replayBufferArray[i])

	end

	return batchArray

end

local function sampleIndex(probabilityArray)

	local randomProbability = math.random()

	local cumulativeProbability = 0

	for i = #probabilityArray, 1, -1 do

		local probability = probabilityArray[i]

		cumulativeProbability = cumulativeProbability + probability

		if (randomProbability >= cumulativeProbability) then continue end

		return i, probability

	end

end


function ExperienceReplay.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewExperienceReplay = {}

	setmetatable(NewExperienceReplay, ExperienceReplay)

	NewExperienceReplay.numberOfRunsToUpdate = parameterDictionary.numberOfRunsToUpdate or parameterDictionary[1] or 1

	NewExperienceReplay.maximumBufferSize = parameterDictionary.maximumBufferSize or parameterDictionary[2] or 100

	NewExperienceReplay.numberOfRuns = parameterDictionary.numberOfRuns or parameterDictionary[3] or 0
	
	NewExperienceReplay.RunFunction = parameterDictionary.RunFunction or parameterDictionary[4]
	
	NewExperienceReplay.AddTemporalDifferenceErrorFunction = parameterDictionary.AddTemporalDifferenceErrorFunction or parameterDictionary[5]
	
	NewExperienceReplay.ResetFunction = parameterDictionary.ResetFunction or parameterDictionary[6]

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
		
		local replayBufferBatchArray = sampleBuffer(replayBufferArray, batchSize)

		for _, experience in ipairs(replayBufferBatchArray) do UpdateFunction(table.unpack(experience)) end
		
	end
	
	return ExperienceReplay.new({numberOfRunsToUpdate, maximumBufferSize, numberOfRuns, RunFunction})
	
end


function ExperienceReplay.NStepExperienceReplay(parameterDictionary)

	parameterDictionary = parameterDictionary or {}
	
	local batchSize = parameterDictionary.batchSize or parameterDictionary[1] or defaultBatchSize

	local numberOfRunsToUpdate = parameterDictionary.numberOfRunsToUpdate or parameterDictionary[2] 

	local maximumBufferSize = parameterDictionary.maximumBufferSize or parameterDictionary[3]
	
	local nStep = parameterDictionary.nStep or parameterDictionary[4] or 3

	local numberOfRuns = parameterDictionary.numberOfRuns or parameterDictionary[5] 

	local replayBufferArray = parameterDictionary.replayBufferArray or parameterDictionary[6] or {}

	local RunFunction = function(UpdateFunction)

		local replayBufferBatchArray = sampleBuffer(replayBufferArray, batchSize)

		local replayBufferArraySize = #replayBufferArray

		local replayBufferBatchArraySize = #replayBufferBatchArray

		local nStep = math.min(nStep, replayBufferBatchArraySize)

		local finalBatchArrayIndex = (replayBufferBatchArraySize - nStep) + 1

		for i = replayBufferBatchArraySize, finalBatchArrayIndex, -1 do UpdateFunction(table.unpack(replayBufferBatchArray[i])) end

	end

	return ExperienceReplay.new({numberOfRunsToUpdate, maximumBufferSize, numberOfRuns, RunFunction})

end

function ExperienceReplay.PrioritizedExperienceReplay(parameterDictionary)

	parameterDictionary = parameterDictionary or {}
	
	local Model = parameterDictionary.Model
	
	if (not Model) then error("No Model!") end
	
	local batchSize = parameterDictionary.batchSize or parameterDictionary[1] or defaultBatchSize
	
	local numberOfRunsToUpdate = parameterDictionary.numberOfRunsToUpdate or parameterDictionary[3] 

	local maximumBufferSize = parameterDictionary.maximumBufferSize or parameterDictionary[4]

	local alpha = parameterDictionary.alpha or parameterDictionary[2] or 0.6

	local beta = parameterDictionary.beta or parameterDictionary[3] or 0.4

	local epsilon = parameterDictionary.epsilon or parameterDictionary[4] or 1e-16
	
	local aggregateFunction = parameterDictionary.aggregateFunction or "Maximum"
	
	local numberOfRuns = parameterDictionary.numberOfRuns or parameterDictionary[5]

	local replayBufferArray = parameterDictionary.replayBufferArray or {}

	local temporalDifferenceArray = parameterDictionary.temporalDifferenceErrorArray or {}

	local priorityArray = parameterDictionary.priorityArray or {}

	local weightArray = parameterDictionary.weightArray or {}
	
	local aggregateFunctionToApply = aggregrateFunctionList[aggregateFunction]
	
	local RunFunction = function(UpdateFunction)

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

			local index, probability = sampleIndex(probabilityArray, sumPriorityAlpha)

			local experience = replayBufferArray[index]

			local temporalDifferenceErrorValueOrVector = temporalDifferenceArray[index]

			local importanceSamplingWeight = math.pow((lowestNumberOfBatchSize * probability), -beta) / math.max(table.unpack(weightArray), epsilon) 

			if (type(temporalDifferenceErrorValueOrVector) ~= "number") then

				temporalDifferenceErrorValueOrVector = aggregateFunctionToApply(temporalDifferenceErrorValueOrVector)

			end

			weightArray[index] = importanceSamplingWeight

			priorityArray[index] = math.abs(temporalDifferenceErrorValueOrVector)

			local outputMatrix = Model{replayBufferArray[i][1]}

			local lossMatrix = outputMatrix * temporalDifferenceErrorValueOrVector * importanceSamplingWeight

			if (sumLossMatrix) then

				sumLossMatrix = sumLossMatrix + lossMatrix

			else

				sumLossMatrix = lossMatrix

			end

		end

		sumLossMatrix:differentiate()

	end

	local AddTemporalDifferenceErrorFunction = function()

		local maximumPriority = 1

		for i, priority in ipairs(priorityArray) do

			if (priority > maximumPriority) then
				
				maximumPriority = priority
				
			end

		end

		table.insert(priorityArray, maximumPriority)

		table.insert(weightArray, 0)

		removeFirstValueFromArrayIfExceedsBufferSize(priorityArray)

		removeFirstValueFromArrayIfExceedsBufferSize(weightArray)

	end
	
	local ResetFunction = function()

		table.clear(priorityArray)

		table.clear(weightArray)

	end

	return ExperienceReplay.new({numberOfRunsToUpdate, maximumBufferSize, numberOfRuns, AddTemporalDifferenceErrorFunction, ResetFunction})

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

	if (numberOfRuns < self.numberOfRunsToUpdate) then return end

	self.numberOfRuns = 0

	self.RunFunction(updateFunction)

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

	removeFirstValueFromArrayIfExceedsBufferSize(replayBufferArray, self.maximumBufferSize)

end

function ExperienceReplay:addTemporalDifferenceError(temporalDifferenceErrorVectorOrValue)
	
	local AddTemporalDifferenceErrorFunction = self.AddTemporalDifferenceErrorFunction

	if (not AddTemporalDifferenceErrorFunction) then return end
	
	local temporalDifferenceErrorArray = self.temporalDifferenceErrorArray

	table.insert(temporalDifferenceErrorArray, temporalDifferenceErrorVectorOrValue)

	if (AddTemporalDifferenceErrorFunction) then AddTemporalDifferenceErrorFunction(temporalDifferenceErrorVectorOrValue) end

	removeFirstValueFromArrayIfExceedsBufferSize(temporalDifferenceErrorArray, self.maximumBufferSize)

end

return ExperienceReplay
