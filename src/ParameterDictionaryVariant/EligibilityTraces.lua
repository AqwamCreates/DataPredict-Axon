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

local EligibilityTrace = {}

EligibilityTrace.__index = EligibilityTrace

function EligibilityTrace.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewEligibilityTrace = {}

	setmetatable(NewEligibilityTrace, EligibilityTrace)
	
	NewEligibilityTrace.incrementFunction = parameterDictionary.incrementFunction or parameterDictionary[1]
	
	NewEligibilityTrace.lambda = parameterDictionary.lambda or parameterDictionary[2] or 0.5

	NewEligibilityTrace.eligibilityTraceTensor = nil
	
	NewEligibilityTrace.isAnObject = true
	
	return NewEligibilityTrace
	
end

function EligibilityTrace.AccumulatingTrace(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local lambda = parameterDictionary.lambda or parameterDictionary[1]
	
	local incrementFunction = function(eligibilityTraceTensor, actionIndex)

			eligibilityTraceTensor[actionIndex] = eligibilityTraceTensor[actionIndex] + 1

			return eligibilityTraceTensor

		end

	return EligibilityTrace.new({incrementFunction, lambda})
	
end

function EligibilityTrace.DutchTrace(parameterDictionary)

	parameterDictionary = parameterDictionary or {}
	
	local lambda = parameterDictionary.lambda or parameterDictionary[1]

	local alpha = parameterDictionary.alpha or parameterDictionary[2] or 0.5

	local incrementFunction = function(eligibilityTraceTensor, actionIndex)

		eligibilityTraceTensor[actionIndex] = ((1 - alpha) * eligibilityTraceTensor[actionIndex]) + 1

		return eligibilityTraceTensor

	end

	return EligibilityTrace.new({incrementFunction, lambda})

end

function EligibilityTrace.ReplacingTrace(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local lambda = parameterDictionary.lambda or parameterDictionary[1]

	local incrementFunction = function(eligibilityTraceTensor, actionIndex)

		eligibilityTraceTensor[actionIndex] = 1

		return eligibilityTraceTensor

	end

	return EligibilityTrace.new({incrementFunction, lambda})

end

function EligibilityTrace:increment(parameterDictionary)
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	parameterDictionary = parameterDictionary or {}
	
	local actionIndex = parameterDictionary.actionIndex or parameterDictionary[1]
	
	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[2]
	
	local numberOfActions = parameterDictionary.numberOfActions or parameterDictionary[3]

	local eligibilityTraceTensor = self.eligibilityTraceTensor or AqwamTensorLibrary:createTensor({numberOfActions}, 0) 

	eligibilityTraceTensor = AqwamTensorLibrary:multiply(eligibilityTraceTensor, discountFactor * self.lambda)

	eligibilityTraceTensor = self.incrementFunction(eligibilityTraceTensor, actionIndex)
	
	self.eligibilityTraceTensor = eligibilityTraceTensor

end

function EligibilityTrace:calculate(parameterDictionary)
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	local temporalDifferenceError = parameterDictionary.temporalDifferenceError or parameterDictionary[1]
	
	local actionIndex = parameterDictionary.actionIndex or parameterDictionary[2]
	
	local eligibilityTraceTensor = self.eligibilityTraceTensor
	
	temporalDifferenceError = AutomaticDifferentiationTensor:fetchValue{temporalDifferenceError}
	
	if (actionIndex) then
		
		return (temporalDifferenceError * eligibilityTraceTensor[actionIndex])

	else

		return AqwamTensorLibrary:multiply(temporalDifferenceError, eligibilityTraceTensor)

	end
	
end

function EligibilityTrace:getLambda()
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	return self.lambda

end

function EligibilityTrace:setLambda(lambda)
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	self.lambda = lambda

end

function EligibilityTrace:reset()
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	self.eligibilityTraceTensor = nil

end

return EligibilityTrace
