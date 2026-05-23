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

local DevianceFunctions = {}

local defaultAlpha = 0.25

local defaultGamma = 2

local function getNumberOfData(value)
	
	if (type(value) == "table") then return #value end
	
	return 1
	
end

local function collapseTensor(tensor, targetDimensionSizeArray)

	local numberOfDimensionsOfTensor = #targetDimensionSizeArray

	local numberOfDimensionsOfDerivativeTensor = #AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensionsToSum = numberOfDimensionsOfDerivativeTensor - numberOfDimensionsOfTensor

	for i = 1, numberOfDimensionsToSum, 1 do tensor = AqwamTensorLibrary:sum(tensor, 1)[1] end

	for i, size in ipairs(targetDimensionSizeArray) do

		if (size == 1) then tensor = AqwamTensorLibrary:sum(tensor, i) end

	end

	return tensor

end

function DevianceFunctions.FastMeanPoissonDeviance(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local inputTensorArray = {generatedLabelTensor, labelTensor}
	
	local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}
	
	local functionToApply = function (h, y) return (y * math.log(y / h) - y + h) end
	
	local poissonDevianceTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor)

	local sumPoissonDevianceTensorValue = AqwamTensorLibrary:sum(poissonDevianceTensor)

	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = (2 * sumPoissonDevianceTensorValue) / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local generatedLabelTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then

			if (generatedLabelTensor:getIsFirstDerivativeTensorRequired()) then

				local functionToApply = function (h, y) return (2 * (1 - (y / h))) end

				local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

			if (labelTensor:getIsFirstDerivativeTensorRequired()) then

				local functionToApply = function (h, y) return (2 * math.log(y / h)) end

				local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

function DevianceFunctions.FastNegativeBinomialDeviance(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local dispersion = parameterDictionary.dispersion or 1
	
	local theta = 1 / dispersion

	local inputTensorArray = {generatedLabelTensor, labelTensor}

	local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}
	
	local functionToApply = function (h, y)
		
		local valuePart1 = (y + theta) / (h + theta)
		
		local valuePart2 = math.log(valuePart1)
		
		local valuePart3 = y / h
		
		local valuePart4 = math.log(valuePart3)
		
		local value = (y * valuePart4) - ((y + theta) * valuePart2)
		
		return value
		
	end
	
	local negativeBinomialDevianceTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor)
	
	local sumNegativeBinomialDevianceTensorValue = AqwamTensorLibrary:sum(negativeBinomialDevianceTensor)

	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = (2 * sumNegativeBinomialDevianceTensorValue) / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local generatedLabelTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then

			if (generatedLabelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local functionToApply = function (h, y)

					local valuePart1 = y / h

					local valuePart2 = (y + theta) / (h + theta)

					local value = 2 * (valuePart2 - valuePart1)

					return value

				end
				
				local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor) 

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

			if (labelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local functionToApply = function (h, y)

					local valuePart1 = y / h

					local valuePart2 = (y + theta) / (h + theta)

					local value = 2 * (math.log(valuePart2) - math.log(valuePart1))
					
					return value

				end

				local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor) 

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

function DevianceFunctions.FastGammaDeviance(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local inputTensorArray = {generatedLabelTensor, labelTensor}
	
	local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}
	
	local functionToApply = function (h, y)
		
		local ratio = y / h

		return (math.log(ratio) + ratio - 1)
		
	end
	
	local gammaDevianceTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor)

	local sumGammaDevianceValue = AqwamTensorLibrary:sum(gammaDevianceTensor)

	local numberOfData = getNumberOfData(labelTensor)
	
	local resultValue = (2 * sumGammaDevianceValue) / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local generatedLabelTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then

			if (generatedLabelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local functionToApply = function (h, y) return (2 * ((h - y) / math.pow(h, 2))) end
				
				local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

			if (labelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local functionToApply = function (h, y) return (2 * ((1 / y) + (1 / h))) end

				local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

------------------------------------------

function DevianceFunctions.MeanPoissonDeviance(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local errorTensor = generatedLabelTensor - labelTensor
	
	local ratioTensor = labelTensor / generatedLabelTensor
	
	local logRatioTensor = AutomaticDifferentiationTensor.logarithm{ratioTensor}
	
	local poissonDevianceTensor = (labelTensor * logRatioTensor) - errorTensor
	
	local sumPoissonDevianceTensorValue = poissonDevianceTensor:sum()

	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = (2 * sumPoissonDevianceTensorValue) / numberOfData

	return resultValue

end

function DevianceFunctions.MeanNegativeBinomialDeviance(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local dispersion = parameterDictionary.dispersion or 1
	
	local theta = 1 / dispersion
	
	local negativeBinomialTensorPart1 = (labelTensor + theta) / (generatedLabelTensor + theta)
	
	local negativeBinomialTensorPart2 = AutomaticDifferentiationTensor.logarithm(negativeBinomialTensorPart1)
	
	local negativeBinomialTensorPart3 = labelTensor / generatedLabelTensor
	
	local negativeBinomialTensorPart4 = AutomaticDifferentiationTensor.logarithm(negativeBinomialTensorPart3)
	
	local negativeBinomialTensor = (labelTensor * negativeBinomialTensorPart4) - ((labelTensor + theta) * negativeBinomialTensorPart2)
	
	local sumNegativeBinomialDevianceTensorValue = negativeBinomialTensor:sum()
	
	local numberOfData = getNumberOfData(labelTensor)
	
	local resultValue = (2 * sumNegativeBinomialDevianceTensorValue) / numberOfData

	return resultValue

end

function DevianceFunctions.MeanGammaDeviance(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]

	local ratioTensor = labelTensor / generatedLabelTensor

	local logRatioTensor = AutomaticDifferentiationTensor.logarithm{ratioTensor}
	
	local gammaDevianceTensor = logRatioTensor + ratioTensor - 1

	local sumGammaDevianceTensorValue = gammaDevianceTensor:sum()

	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = (2 * sumGammaDevianceTensorValue) / numberOfData

	return resultValue

end

return DevianceFunctions
