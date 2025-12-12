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

local LinkLayer = {}

local epsilon = 1e-16

local epsilonComplement = 1 - epsilon

function LinkLayer.FastLogit(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function(z) return math.log(z / (1 - z)) end

	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}
	
	local clampedTensor = AqwamTensorLibrary:applyFunction(math.clamp, pureTensor, epsilon, epsilonComplement)

	local resultTensor = AqwamTensorLibrary:applyFunction(functionToApply, clampedTensor)

	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end

			local functionToApply = function (z) return 1 / (z * (1 - z)) end

			local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, clampedTensor)

			tensor:differentiate{AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)}

		end

	end

	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function LinkLayer.FastInverseLogit(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function(z) return 1/(1 + math.exp(-z)) end
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureTensor)
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end
			
			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end

			local functionToApply = function (a) return (a * (1 - a)) end

			local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureTensor)

			tensor:differentiate{AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)}

		end
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function LinkLayer.FastLogLog(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function(z) return math.log(-math.log(z)) end

	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}
	
	local clampedTensor = AqwamTensorLibrary:applyFunction(math.clamp, pureTensor, epsilon, 1)

	local resultTensor = AqwamTensorLibrary:applyFunction(functionToApply, clampedTensor)

	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end

			local functionToApply = function (z) return 1 / -math.log(z) end

			local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, clampedTensor)

			tensor:differentiate{AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)}

		end

	end

	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function LinkLayer.FastInverseLogLog(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function(z) return math.exp(-math.exp(z)) end

	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureTensor)

	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end

			local functionToApply = function (z) return -math.exp(z) * math.exp(-math.exp(z)) end

			local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureTensor)

			tensor:differentiate{AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)}

		end

	end

	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function LinkLayer.FastComplementLogLog(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function(z) return math.log(-math.log(1 - z)) end

	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}
	
	local clampedTensor = AqwamTensorLibrary:applyFunction(math.clamp, pureTensor, 0, epsilonComplement)

	local resultTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureTensor)

	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end

			local functionToApply = function (z) return 1 / ((1 - z) * math.log((1 - z))) end

			local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, clampedTensor)

			tensor:differentiate{AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)}

		end

	end

	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function LinkLayer.FastInverseComplementLogLog(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function(z) return 1 - math.exp(-math.exp(z)) end

	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureTensor)

	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end

			local functionToApply = function (z) return math.exp(z) * math.exp(-math.exp(z)) end

			local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureTensor)

			tensor:differentiate{AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)}

		end

	end

	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function LinkLayer.FastPoisson(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function(z) return math.log(z) end

	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}
	
	local clampedTensor = AqwamTensorLibrary:applyFunction(math.clamp, pureTensor, epsilon, 1)

	local resultTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureTensor)

	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end

			local functionToApply = function (z) return 1 / z end

			local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, clampedTensor)

			tensor:differentiate{AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)}

		end

	end

	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function LinkLayer.FastInversePoisson(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local functionToApply = function(z) return math.exp(z) end

	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureTensor)

	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end
			
			-- Note: Derivative of exponent(z) is exponent(z), where a = exponent(z). Therefore, we're taking a shortcut to reduce computational resources.

			tensor:differentiate{AqwamTensorLibrary:multiply(firstDerivativeTensor, resultTensor)}

		end

	end

	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {tensor}})

end

function LinkLayer.Logit(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]
	
	local xTensor = AutomaticDifferentiationTensor.clamp{tensor, epsilon, epsilonComplement}

	local resultTensorPart1 = xTensor / (1 - xTensor)

	return AutomaticDifferentiationTensor.logarithm{resultTensorPart1}

end

function LinkLayer.InverseLogit(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local exponentTensor = AutomaticDifferentiationTensor.exponent{-tensor}

	return 1 / (1 + exponentTensor)

end

function LinkLayer.LogLog(parameterDictionary)
	
	local tensor = parameterDictionary.tensor or parameterDictionary[1]
	
	local clampedTensor = AutomaticDifferentiationTensor.clamp{tensor, epsilon, 1}

	return AutomaticDifferentiationTensor.logarithm{-AutomaticDifferentiationTensor.logarithm{clampedTensor}}

end

function LinkLayer.InverseLogLog(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	return AutomaticDifferentiationTensor.exponent{-AutomaticDifferentiationTensor.exponent{tensor}}

end

function LinkLayer.ComplementaryLogLog(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local clampedTensor = AutomaticDifferentiationTensor.clamp{tensor, 0, epsilonComplement}

	return AutomaticDifferentiationTensor.logarithm{-AutomaticDifferentiationTensor.logarithm{1 - clampedTensor}}

end

function LinkLayer.InverseComplementaryLogLog(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local resultTensor = 1 - AutomaticDifferentiationTensor.exponent{-AutomaticDifferentiationTensor.exponent{tensor}}

	return resultTensor

end

function LinkLayer.Poisson(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local maximumTensor = AutomaticDifferentiationTensor.maximum{tensor, epsilon}

	return AutomaticDifferentiationTensor.logarithm{maximumTensor}

end

function LinkLayer.InversePoisson(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	return AutomaticDifferentiationTensor.exponent{tensor}

end

return LinkLayer
