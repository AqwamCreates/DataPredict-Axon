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

local ActivationFunctions = {}

local function sumToChainRuleFirstDerivativeTensorWhenSameDimensionIndex(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, targetTensor)

	local nextDimension = currentDimension + 1

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			sumToChainRuleFirstDerivativeTensorWhenSameDimensionIndex(tensor[i], dimensionSizeArray, numberOfDimensions, nextDimension, targetTensor[i])

		end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			local value = tensor[i]

			targetTensor[i] = targetTensor[i] + (value * (1 - value))

		end

	end

end

local function sumToChainRuleFirstDerivativeTensorWhenDifferentDimensionIndex(tensor1, tensor2, dimensionSizeArray, numberOfDimensions, currentDimension, targetTensor)

	local nextDimension = currentDimension + 1

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			sumToChainRuleFirstDerivativeTensorWhenDifferentDimensionIndex(tensor1[i], tensor2[i], dimensionSizeArray, numberOfDimensions, nextDimension, targetTensor[i])

		end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			local calculatedTensor = tensor1[i] * (1 - tensor2[i])

			targetTensor[i] = targetTensor[i] + calculatedTensor

		end

	end

end

local function calculateChainRuleFirstDerivativeTensor(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, targetTensor, sumDimension)

	local nextDimension = currentDimension + 1

	if (currentDimension < (sumDimension - 1)) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			calculateChainRuleFirstDerivativeTensor(tensor[i], dimensionSizeArray, numberOfDimensions, nextDimension, targetTensor[i], sumDimension)

		end

	else

		for i, subTensor1 in ipairs(tensor) do

			for j, subTensor2 in ipairs(tensor) do

				if (i == j) then

					sumToChainRuleFirstDerivativeTensorWhenSameDimensionIndex(subTensor1, dimensionSizeArray, numberOfDimensions, nextDimension, targetTensor[i])

				else

					sumToChainRuleFirstDerivativeTensorWhenDifferentDimensionIndex(subTensor1, subTensor2, dimensionSizeArray, numberOfDimensions, nextDimension, targetTensor[i])

				end

			end

		end

	end	

end

function ActivationFunctions.FastSigmoid(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function(z) return 1/(1 + math.exp(-1 * z)) end

	local resultTensor = AqwamTensorLibrary:applyFunction(tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local functionToApply = function (a) return (a * (1 - a)) end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, tensor)

		tensor:differentiate{chainRuleFirstDerivativeTensor}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function ActivationFunctions.FastBinaryStep(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function (z) return ((z > 0) and 1) or 0 end

	local resultTensor =  AqwamTensorLibrary:applyFunction(functionToApply, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		tensor:differentiate(AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0))

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function ActivationFunctions.FastRectifiedLinearUnit(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function (z) return math.max(z, 0) end

	local resultTensor =  AqwamTensorLibrary:applyFunction(functionToApply, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local derivativeFunctionToApply = function (z) if (z >= 0) then return 1 else return 0 end end

		local gradientTensor = AqwamTensorLibrary:applyFunction(derivativeFunctionToApply, tensor)

		tensor:differentiate{AqwamTensorLibrary:multiply(gradientTensor, firstDerivativeTensor)}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function ActivationFunctions.FastLeakyRectifiedLinearUnit(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local negativeSlopeFactor = parameterDictionary.negativeSlopeFactor or parameterDictionary[2] or 0.01

	local functionToApply = function (z) return math.max(z, z * negativeSlopeFactor) end

	local resultTensor =  AqwamTensorLibrary:applyFunction(functionToApply, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local partialDerivativeFunctionToApply = function (z) if (z >= 0) then return 1 else return negativeSlopeFactor end end

		local gradientTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate{AqwamTensorLibrary:multiply(gradientTensor, firstDerivativeTensor)}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function ActivationFunctions.FastExponentLinearUnit(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local negativeSlopeFactor = parameterDictionary.negativeSlopeFactor or parameterDictionary[2] or 0.01

	local functionToApply = function (z) return if (z > 0) then z else ((math.exp(z) - 1) * negativeSlopeFactor) end

	local resultTensor =  AqwamTensorLibrary:applyFunction(functionToApply, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local partialDerivativeFunctionToApply = function (z) if (z > 0) then return 1 else return (negativeSlopeFactor * math.exp(z)) end end

		local gradientTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate{AqwamTensorLibrary:multiply(gradientTensor, firstDerivativeTensor)}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function ActivationFunctions.FastSigmoidLinearUnit(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function (z) return z / (1 + math.exp(-z)) end

	local resultTensor =  AqwamTensorLibrary:applyFunction(functionToApply, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local partialDerivativeFunctionToApply = function (z) return (1 + math.exp(-z) + (z * math.exp(-z))) / (1 + math.exp(-z))^2 end

		local gradientTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate{AqwamTensorLibrary:multiply(gradientTensor, firstDerivativeTensor)}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function ActivationFunctions.FastGaussian(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function (z) return math.exp(-math.pow(z, 2)) end

	local resultTensor =  AqwamTensorLibrary:applyFunction(functionToApply, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local partialDerivativeFunctionToApply = function (z) return -2 * z * math.exp(-math.pow(z, 2)) end

		local gradientTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate{AqwamTensorLibrary:multiply(gradientTensor, firstDerivativeTensor)}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function ActivationFunctions.FastMish(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function (z) return (z * math.tanh(math.log(1 + math.exp(z)))) end

	local resultTensor =  AqwamTensorLibrary:applyFunction(functionToApply, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local partialDerivativeFunctionToApply = function (z) return (math.exp(z) * (math.exp(3 * z) + 4 * math.exp(2 * z) + (6 + 4 * z) * math.exp(z) + 4 * (1 + z)) / math.pow((1 + math.pow((math.exp(z) + 1), 2)), 2)) end

		local gradientTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate{AqwamTensorLibrary:multiply(gradientTensor, firstDerivativeTensor)}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function ActivationFunctions.FastTanh(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local resultTensor =  AqwamTensorLibrary:applyFunction(math.tanh, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local partialDerivativeFunctionToApply = function (a) return (1 - math.pow(a, 2)) end

		local gradientTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate{AqwamTensorLibrary:multiply(gradientTensor, firstDerivativeTensor)}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function ActivationFunctions.FastSoftmax(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local dimension = parameterDictionary.dimension or parameterDictionary[2] or 1

	local exponentTensor = AqwamTensorLibrary:applyFunction(math.exp, tensor)

	local sumExponentTensor = AqwamTensorLibrary:sum(exponentTensor, dimension)

	local resultTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentTensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		print(not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor))

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

		local gradientTensor = AqwamTensorLibrary:createTensor(dimensionSizeArray)

		calculateChainRuleFirstDerivativeTensor(resultTensor, dimensionSizeArray, #dimensionSizeArray, 1, gradientTensor, dimension)

		tensor:differentiate{AqwamTensorLibrary:multiply(gradientTensor, firstDerivativeTensor)}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function ActivationFunctions.Sigmoid(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local exponentTensor = AutomaticDifferentiationTensor.exponent{-1 * tensor}

	return 1 / (1 + exponentTensor)

end

function ActivationFunctions.BinaryStep(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function (z) return ((z > 0) and 1) or 0 end

	local resultTensor =  AqwamTensorLibrary:applyFunction(functionToApply, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		tensor:differentiate(AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0))

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function ActivationFunctions.RectifiedLinearUnit(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	return AutomaticDifferentiationTensor.maximum{tensor, 0}

end

function ActivationFunctions.LeakyRectifiedLinearUnit(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local negativeSlopeFactor = parameterDictionary.negativeSlopeFactor or parameterDictionary[2] or 0.01

	return AutomaticDifferentiationTensor.maximum{tensor, tensor * negativeSlopeFactor}

end

function ActivationFunctions.ExponentLinearUnit(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local negativeSlopeFactor = parameterDictionary.negativeSlopeFactor or parameterDictionary[2] or 0.01

	return AutomaticDifferentiationTensor.maximum{tensor, (AutomaticDifferentiationTensor.exponent{tensor} - 1) * negativeSlopeFactor}

end

function ActivationFunctions.SigmoidLinearUnit(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	return 1 / (1 + AutomaticDifferentiationTensor.exponent{-tensor})

end

function ActivationFunctions.Gaussian(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	return AutomaticDifferentiationTensor.exponent{-(tensor^2)}

end

function ActivationFunctions.Mish(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	return tensor * AutomaticDifferentiationTensor.tanh{AqwamTensorLibrary.log{1 + AqwamTensorLibrary.exponent{tensor}}}

end

function ActivationFunctions.Tanh(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	return AutomaticDifferentiationTensor.tanh(tensor)

end

function ActivationFunctions.Softmax(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local dimension = parameterDictionary.dimension or parameterDictionary[2] or 1

	local exponentTensor = AutomaticDifferentiationTensor.exponent{tensor}

	local sumExponentTensor = exponentTensor:sum{dimension}

	return exponentTensor / sumExponentTensor

end

return ActivationFunctions