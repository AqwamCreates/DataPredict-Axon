function EligibilityTrace:calculate(parameterDictionary)
	
	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	parameterDictionary = parameterDictionary or {}
	
	local temporalDifferenceError = parameterDictionary.temporalDifferenceError or parameterDictionary[1]
	
	local actionIndex = parameterDictionary.actionIndex or parameterDictionary[2]
	
	local discountFactor = parameterDictionary.discountFactor or parameterDictionary[3]
	
	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[4]

	local eligibilityTraceTensor = self.eligibilityTraceTensor or AqwamTensorLibrary:createTensor(dimensionSizeArray, 0) 

	eligibilityTraceTensor = AqwamTensorLibrary:multiply(eligibilityTraceTensor, discountFactor * self.lambda)

	eligibilityTraceTensor = self.IncrementFunction(eligibilityTraceTensor, actionIndex)
	
	self.eligibilityTraceTensor = eligibilityTraceTensor
	
	temporalDifferenceError = AutomaticDifferentiationTensor:fetchValue{temporalDifferenceError}
	
	local isTemporalDifferenceErrorATensor = (type(temporalDifferenceError) == "table")
	
	if (isTemporalDifferenceErrorATensor) then
		
		return AqwamTensorLibrary:multiply(temporalDifferenceError, eligibilityTraceTensor)
		
	else
		
		return (temporalDifferenceError * eligibilityTraceTensor[1][actionIndex])
		
	end

end
