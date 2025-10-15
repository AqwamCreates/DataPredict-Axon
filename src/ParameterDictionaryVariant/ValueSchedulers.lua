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

local ValueScheduler = {}

ValueScheduler.__index = ValueScheduler

local function showFunctionErrorDueToNonObjectCondition(showError)

	if (showError) then error("This function can only be called if it is an object.") end

end

function ValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewValueScheduler = {}

	setmetatable(NewValueScheduler, ValueScheduler)
	
	NewValueScheduler.CalculateFunction = parameterDictionary.CalculateFunction or parameterDictionary[1]
	
	NewValueScheduler.timeValue = parameterDictionary.timeValue or parameterDictionary[2] or 0
	
	NewValueScheduler.isAnObject = true
	
	return NewValueScheduler
	
end

function ValueScheduler.Chained(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local ValueSchedulerArray = parameterDictionary.ValueSchedulerArray or parameterDictionary[1]
	
	local timeValue = parameterDictionary.timeValue or parameterDictionary[2]

	local CalculateFunction = function(value, timeValue)

		for _, ValueScheduler in ipairs(ValueSchedulerArray) do value = ValueScheduler:calculate(value, timeValue) end

		return value

	end

	return ValueScheduler.new({CalculateFunction, timeValue})
	
end

function ValueScheduler.Constant(parameterDictionary)
	
	local timeValue = parameterDictionary.timeValue or parameterDictionary[1] or 1

	local decayRate = parameterDictionary.decayRate or  parameterDictionary[2] or 0.5
	
	local timeValue = parameterDictionary.timeValue or parameterDictionary[3]

	local CalculateFunction = function(value, timeValue)

		if (timeValue <= timeValue) then return value end

		return (value * decayRate)

	end)
	
	return ValueScheduler.new({CalculateFunction, timeValue}) 
	
end

function ValueScheduler.CosineAnnealing(parameterDictionary)

	local maximumTimeValue = parameterDictionary.maximumTimeValue or parameterDictionary[1] or defaultMaximumTimeValue

	local minimumValue = parameterDictionary.minimumValue or parameterDictionary[2] or defaultMinimumValue

	local timeValue = parameterDictionary.timeValue or parameterDictionary[3]

	local CalculateFunction = function(value, timeValue)

		local multiplyValuePart1 = 1 + math.cos((timeValue * math.pi) / maximumTimeValue)

		local multiplyValuePart2 = (value - minimumValue)

		local multiplyValue = 0.5 * multiplyValuePart1 * multiplyValuePart2

		return (minimumValue + multiplyValue)

	end)

	return ValueScheduler.new({CalculateFunction, timeValue}) 

end

function ValueScheduler:calculate(parameterDictionary)
	
	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	parameterDictionary = parameterDictionary or {}
	
	local valueToSchedule = parameterDictionary.valueToSchedule or parameterDictionary[1]
	
	local valueToScale = parameterDictionary.valueToScale or parameterDictionary[2]
	
	local CalculateFunction = self.CalculateFunction
	
	if (not CalculateFunction) then error("No calculate function.") end
	
	local timeValue = self.timeValue + 1
	
	self.timeValue = timeValue
	
	valueToSchedule = CalculateFunction(valueToSchedule, timeValue)
	
	if (not valueToScale) then valueToSchedule end
	
	return AqwamTensorLibrary:multiply(valueToSchedule, valueToScale)
	
end

function ValueScheduler:reset()
	
	self.timeValue = 0
	
end

return ValueScheduler
