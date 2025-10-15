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

local defaultDecayRate = 0.5

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
	
	parameterDictionary = parameterDictionary or {}
	
	local timeValue = parameterDictionary.timeValue or parameterDictionary[1] or 1

	local decayRate = parameterDictionary.decayRate or  parameterDictionary[2] or defaultDecayRate
	
	local timeValue = parameterDictionary.timeValue or parameterDictionary[3]

	local CalculateFunction = function(value, timeValue)

		if (timeValue <= timeValue) then return value end

		return (value * decayRate)

	end
	
	return ValueScheduler.new({CalculateFunction, timeValue}) 
	
end

function ValueScheduler.CosineAnnealing(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local maximumTimeValue = parameterDictionary.maximumTimeValue or parameterDictionary[1] or 5

	local minimumValue = parameterDictionary.minimumValue or parameterDictionary[2] or 1

	local timeValue = parameterDictionary.timeValue or parameterDictionary[3]

	local CalculateFunction = function(value, timeValue)

		local multiplyValuePart1 = 1 + math.cos((timeValue * math.pi) / maximumTimeValue)

		local multiplyValuePart2 = (value - minimumValue)

		local multiplyValue = 0.5 * multiplyValuePart1 * multiplyValuePart2

		return (minimumValue + multiplyValue)

	end

	return ValueScheduler.new({CalculateFunction, timeValue}) 

end

function ValueScheduler.Exponential(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local decayRate = parameterDictionary.decayRate or parameterDictionary[1] or defaultDecayRate

	local timeValue = parameterDictionary.timeValue or parameterDictionary[2]

	local CalculateFunction = function(value, timeValue) return (value * math.exp(-decayRate * timeValue)) end

	return ValueScheduler.new({CalculateFunction, timeValue}) 

end

function ValueScheduler.InverseSquareRoot(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local timeValue = parameterDictionary.timeValue or parameterDictionary[1]

	local CalculateFunction = function(value, timeValue) return (value / math.pow(timeValue, 0.5)) end

	return ValueScheduler.new({CalculateFunction, timeValue})

end

function ValueScheduler.InverseTime(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local decayRate = parameterDictionary.decayRate or parameterDictionary[1] or defaultDecayRate

	local timeValue = parameterDictionary.timeValue or parameterDictionary[2]

	local CalculateFunction = function(value, timeValue) return (value / (1 + (decayRate * timeValue))) end

	return ValueScheduler.new({CalculateFunction, timeValue}) 

end

function ValueScheduler.Linear(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local otherTimeValue = parameterDictionary.timeValue or parameterDictionary[1] or 5
	
	local startFactor = parameterDictionary.startFactor or parameterDictionary[2] or 1/3

	local endFactor = parameterDictionary.endFactor or parameterDictionary[3] or 1
	
	local timeValue = parameterDictionary.otherTimeValue or parameterDictionary[4]

	local CalculateFunction = function(value, timeValue)

		if (timeValue >= otherTimeValue) then return (value * endFactor) end

		local factor = startFactor + ((endFactor - startFactor) * (timeValue / otherTimeValue))

		return (value * factor)

	end

	return ValueScheduler.new({CalculateFunction, timeValue}) 

end

function ValueScheduler.MultipleStep(parameterDictionary)

	parameterDictionary = parameterDictionary or {}
	
	local timeValueArray = parameterDictionary.timeValueArray or parameterDictionary[1]

	local decayRate = parameterDictionary.decayRate or parameterDictionary[2] or defaultDecayRate

	local timeValue = parameterDictionary.timeValue or parameterDictionary[3]

	local CalculateFunction = function(value, timeValue)

		local decayCount = 0

		for i, timeValueMilestone in ipairs(timeValueArray) do

			if (timeValue <= timeValueMilestone) then break end

			decayCount = decayCount + 1

		end

		return (value * math.pow(decayRate, decayCount))

	end

	return ValueScheduler.new({CalculateFunction, timeValue}) 

end

function ValueScheduler.Multiplicative(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local functionToRun = parameterDictionary.functionToRun or parameterDictionary[1]

	local timeValue = parameterDictionary.timeValue or parameterDictionary[2]

	local CalculateFunction = function(value, timeValue) return (value * functionToRun(timeValue)) end

	return ValueScheduler.new({CalculateFunction, timeValue}) 

end

function ValueScheduler.Polynomial(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local totalTimeValue = parameterDictionary.totalTimeValue or parameterDictionary[1] or 5
	
	local power = parameterDictionary.power or parameterDictionary[2] or 1

	local timeValue = parameterDictionary.timeValue or parameterDictionary[3]

	local CalculateFunction = function(value, timeValue) return (value * math.pow((1 - (timeValue / totalTimeValue)), power)) end

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
	
	if (not valueToScale) then return valueToSchedule end
	
	return AqwamTensorLibrary:multiply(valueToSchedule, valueToScale)
	
end

function ValueScheduler:reset()
	
	self.timeValue = 0
	
end

return ValueScheduler
