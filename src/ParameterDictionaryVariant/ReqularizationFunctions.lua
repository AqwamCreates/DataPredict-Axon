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

local ReqularizationFunctions = {}

local defaultLambda = 0.01

function ReqularizationFunctions.FastLasso(parameterDictionary)

	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[1]

	local lambda = parameterDictionary.lambda or parameterDictionary[2] or defaultLambda

	local numberOfData = parameterDictionary.numberOfData or parameterDictionary[3]

	local inputTensorArray = {weightTensor}

	local pureWeightTensor = AutomaticDifferentiationTensor:fetchValue{weightTensor}

	local functionToApply = function (weightValue) return (lambda * math.abs(weightValue)) end

	local absoluteWeightTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureWeightTensor)

	local sumAbsoluteWeightValue = AqwamTensorLibrary:sum(absoluteWeightTensor)

	local resultValue = sumAbsoluteWeightValue / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local weightTensor = inputTensorArray[1]

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{weightTensor}) then return end

		if (not weightTensor:getIsFirstDerivativeTensorRequired()) then return end

		local pureWeightTensor = AutomaticDifferentiationTensor:fetchValue{weightTensor}
		
		local signTensor = AqwamTensorLibrary:applyFunction(math.sign, pureWeightTensor)

		firstDerivativeTensor = AqwamTensorLibrary:multiply(lambda, signTensor, firstDerivativeTensor)
		
		firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

		weightTensor:differentiate{firstDerivativeTensor}

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

function ReqularizationFunctions.FastRidge(parameterDictionary)

	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[1]

	local lambda = parameterDictionary.lambda or parameterDictionary[2] or defaultLambda

	local numberOfData = parameterDictionary.numberOfData or parameterDictionary[3]
	
	local inputTensorArray = {weightTensor}
	
	local pureWeightTensor = AutomaticDifferentiationTensor:fetchValue{weightTensor}

	local functionToApply = function (weightValue) return (lambda * math.pow(weightValue, 2)) end

	local squaredWeightTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureWeightTensor)

	local sumSquaredWeightValue = AqwamTensorLibrary:sum(squaredWeightTensor)

	local resultValue = sumSquaredWeightValue / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local weightTensor = inputTensorArray[1]

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{weightTensor}) then return end

		if (not weightTensor:getIsFirstDerivativeTensorRequired()) then return end
		
		local pureWeightTensor = AutomaticDifferentiationTensor:fetchValue{weightTensor}

		firstDerivativeTensor = AqwamTensorLibrary:multiply((2 * lambda), pureWeightTensor, firstDerivativeTensor)
		
		firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

		weightTensor:differentiate{firstDerivativeTensor}

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

function ReqularizationFunctions.FastElasticNet(parameterDictionary)

	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[1]

	local lambda = parameterDictionary.lambda or parameterDictionary[2] or defaultLambda

	local numberOfData = parameterDictionary.numberOfData or parameterDictionary[3]

	local inputTensorArray = {weightTensor}

	local pureWeightTensor = AutomaticDifferentiationTensor:fetchValue{weightTensor}

	local functionToApply = function (weightValue) return (lambda * (math.pow(weightValue, 2) + math.abs(weightValue))) end

	local elasticNetWeightTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureWeightTensor)

	local sumElasticNetWeightValue = AqwamTensorLibrary:sum(elasticNetWeightTensor)

	local resultValue = sumElasticNetWeightValue / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local weightTensor = inputTensorArray[1]

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{weightTensor}) then return end

		if (not weightTensor:getIsFirstDerivativeTensorRequired()) then return end

		local pureWeightTensor = AutomaticDifferentiationTensor:fetchValue{weightTensor}
		
		local signTensor = AqwamTensorLibrary:applyFunction(math.sign, pureWeightTensor)
		
		local firstDerivativeTensorPart1 = AqwamTensorLibrary:multiply(2, weightTensor)
		
		local firstDerivativeTensorPart2 = AqwamTensorLibrary:add(firstDerivativeTensorPart1, signTensor)
		
		firstDerivativeTensor = AqwamTensorLibrary:multiply(lambda, firstDerivativeTensorPart2, firstDerivativeTensor)

		return AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

------------------------------------------

function ReqularizationFunctions.Lasso(parameterDictionary)

	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[1]

	local lambda = parameterDictionary.lambda or parameterDictionary[2] or defaultLambda

	local numberOfData = parameterDictionary.numberOfData or parameterDictionary[3]

	local absoluteWeightTensor = lambda * weightTensor:absolute()

	local sumAbsoluteWeightValue = absoluteWeightTensor:sum()

	local resultValue = sumAbsoluteWeightValue / numberOfData

	return resultValue

end

function ReqularizationFunctions.Ridge(parameterDictionary)

	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[1]

	local lambda = parameterDictionary.lambda or parameterDictionary[2] or defaultLambda
	
	local numberOfData = parameterDictionary.numberOfData or parameterDictionary[3]
	
	local absoluteWeightTensor = lambda * weightTensor^2

	local sumSquaredWeightValue = absoluteWeightTensor:sum()

	local resultValue = sumSquaredWeightValue / numberOfData

	return resultValue

end

function ReqularizationFunctions.ElasticNet(parameterDictionary)

	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[1]

	local lambda = parameterDictionary.lambda or parameterDictionary[2] or defaultLambda

	local numberOfData = parameterDictionary.numberOfData or parameterDictionary[3]

	local elasticNetWeightTensor = lambda * (weightTensor^2 + weightTensor:absolute())

	local sumElasticNetWeightValue = elasticNetWeightTensor:sum()

	local resultValue = sumElasticNetWeightValue / numberOfData

	return resultValue

end

return ReqularizationFunctions
