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

local EligibilityTrace = {}

EligibilityTrace.__index = EligibilityTrace

local function showFunctionErrorDueToNonObjectCondition(showError)

	if (showError) then error("This function can only be called if it is an object.") end

end

function EligibilityTrace.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewEligibilityTrace = {}

	setmetatable(NewEligibilityTrace, EligibilityTrace)
	
	NewEligibilityTrace.lambda = parameterDictionary.lambda or defaultLambda

	NewEligibilityTrace.mode = parameterDictionary.mode or defaultMode
	
	NewEligibilityTrace.IncrementFunction = parameterDictionary.IncrementFunction or parameterDictionary[1]

	NewEligibilityTrace.eligibilityTraceMatrix = nil
	
	NewEligibilityTrace.isAnObject = true
	
	return NewEligibilityTrace
	
end

function EligibilityTrace.AccumulatingTrace()

	local IncrementFunction = function(eligibilityTraceTensor, actionIndex)
		
		eligibilityTraceTensor[1][actionIndex] = eligibilityTraceTensor[1][actionIndex] + 1

		return eligibilityTraceTensor

	end

	return EligibilityTrace.new({IncrementFunction})
	
end

function EligibilityTrace.DutchTrace(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local alpha = parameterDictionary.alpha or parameterDictionary[1] or 100

	local IncrementFunction = function(eligibilityTraceTensor, actionIndex)

		eligibilityTraceTensor[1][actionIndex] = ((1 - NewDutchTrace.alpha) * eligibilityTraceTensor[1][actionIndex]) + 1

		return eligibilityTraceTensor

	end

	return EligibilityTrace.new({IncrementFunction})

end

function EligibilityTrace.ReplacingTrace()

	local IncrementFunction = function(eligibilityTraceTensor, actionIndex)

		eligibilityTraceTensor[1][actionIndex] = 1

		return eligibilityTraceTensor

	end

	return EligibilityTrace.new({IncrementFunction})

end

function EligibilityTrace:increment(parameterDictionary)
	
	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	parameterDictionary = parameterDictionary or {}
	
	local actionIndex = parameterDictionary.actionIndex or parameterDictionary[1]
	
	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[2]
	
	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[3]

	local eligibilityTraceTensor = self.eligibilityTraceTensor or AqwamTensorLibrary:createTensor(dimensionSizeArray, 0) 

	eligibilityTraceTensor = AqwamTensorLibrary:multiply(eligibilityTraceTensor, discountFactor * self.lambda)

	self.eligibilityTraceTensor = self.IncrementFunction(eligibilityTraceTensor, actionIndex)

end

function EligibilityTrace:calculate(parameterDictionary)
	
	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	parameterDictionary = parameterDictionary or {}
	
	temporalDifferenceErrorVector = parameterDictionary.temporalDifferenceErrorVector or parameterDictionary[1]

	return AqwamTensorLibrary:multiply(temporalDifferenceErrorVector, self.eligibilityTraceTensor)

end

function EligibilityTrace:getLambda()
	
	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	return self.lambda

end

function BaseEligibilityTrace:setLambda(lambda)
	
	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	self.lambda = lambda

end

function BaseEligibilityTrace:reset()
	
	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	self.eligibilityTraceTensor = nil

end

return EligibilityTrace
