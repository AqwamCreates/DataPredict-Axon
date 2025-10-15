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

local EligibilityTrace = {}

EligibilityTrace.__index = EligibilityTrace

local function showFunctionErrorDueToNonObjectCondition(showError)

	if (showError) then error("This function can only be called if it is an object.") end

end

function EligibilityTrace.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewEligibilityTrace = {}

	setmetatable(NewEligibilityTrace, EligibilityTrace)
	
	NewEligibilityTrace.IncrementFunction = parameterDictionary.IncrementFunction or parameterDictionary[1]
	
	NewEligibilityTrace.lambda = parameterDictionary.lambda or parameterDictionary[2] or 0.5

	NewEligibilityTrace.eligibilityTraceTensor = nil
	
	NewEligibilityTrace.isAnObject = true
	
	return NewEligibilityTrace
	
end

function EligibilityTrace.AccumulatingTrace(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.IncrementFunction = function(eligibilityTraceTensor, actionIndex)

			eligibilityTraceTensor[1][actionIndex] = eligibilityTraceTensor[1][actionIndex] + 1

			return eligibilityTraceTensor

		end

	return EligibilityTrace.new(parameterDictionary)
	
end

function EligibilityTrace.DutchTrace(parameterDictionary)

	parameterDictionary = parameterDictionary or {}
	
	local lambda = parameterDictionary.lambda or parameterDictionary[1]

	local alpha = parameterDictionary.alpha or parameterDictionary[2] or 0.5

	local IncrementFunction = function(eligibilityTraceTensor, actionIndex)

		eligibilityTraceTensor[1][actionIndex] = ((1 - alpha) * eligibilityTraceTensor[1][actionIndex]) + 1

		return eligibilityTraceTensor

	end

	return EligibilityTrace.new({IncrementFunction, lambda})

end

function EligibilityTrace.ReplacingTrace()
	
	parameterDictionary = parameterDictionary or {}

	parameterDictionary.IncrementFunction = function(eligibilityTraceTensor, actionIndex)

		eligibilityTraceTensor[1][actionIndex] = 1

		return eligibilityTraceTensor

	end

	return EligibilityTrace.new(parameterDictionary)

end

function EligibilityTrace:calculate(parameterDictionary)
	
	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	parameterDictionary = parameterDictionary or {}
	
	local temporalDifferenceError = parameterDictionary.temporalDifferenceError or parameterDictionary[1]
	
	local actionIndex = parameterDictionary.actionIndex or parameterDictionary[2]
	
	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[3]
	
	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[4]

	local eligibilityTraceTensor = self.eligibilityTraceTensor or AqwamTensorLibrary:createTensor(dimensionSizeArray, 0) 

	eligibilityTraceTensor = AqwamTensorLibrary:multiply(eligibilityTraceTensor, discountFactor * self.lambda)

	self.eligibilityTraceTensor = self.IncrementFunction(eligibilityTraceTensor, actionIndex)
	
	temporalDifferenceError = AutomaticDifferentiationTensor:fetchValue{temporalDifferenceError}
	
	local isTemporalDifferenceErrorATensor = (type(temporalDifferenceError) == "table")
	
	if (isTemporalDifferenceErrorATensor) then
		
		return AqwamTensorLibrary:multiply(temporalDifferenceError, eligibilityTraceTensor)
		
	else
		
		return (temporalDifferenceError * eligibilityTraceTensor[1][actionIndex])
		
	end

end

function EligibilityTrace:getLambda()
	
	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	return self.lambda

end

function EligibilityTrace:setLambda(lambda)
	
	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	self.lambda = lambda

end

function EligibilityTrace:reset()
	
	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	self.eligibilityTraceTensor = nil

end

return EligibilityTrace
