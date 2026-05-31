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

local FusedCostFunctions = {}

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

local function subtractMinusOneAtTopPositionIndexFromTensor(partialFirstDerivativeTensor, labelTensor, dimensionSizeArray, numberOfDimensions, currentDimension, topPositionIndex)
	
	if (numberOfDimensions == 0) then 
		
		if (topPositionIndex == labelTensor) then return (partialFirstDerivativeTensor - 1) end
		
		return partialFirstDerivativeTensor
		
	end
	
	local currentDimensionSize = dimensionSizeArray[currentDimension]
	
	if (currentDimension < numberOfDimensions) then
		
		local nextDimension = currentDimension + 1
		
		for i = 1, currentDimensionSize, 1 do

			partialFirstDerivativeTensor[i] = subtractMinusOneAtTopPositionIndexFromTensor(partialFirstDerivativeTensor[i], labelTensor[i], dimensionSizeArray, numberOfDimensions, nextDimension, topPositionIndex)

		end
		
	else
		
		if (currentDimensionSize) then
			
			for i = 1, currentDimensionSize, 1 do

				if (topPositionIndex == labelTensor[i]) then partialFirstDerivativeTensor[i] = partialFirstDerivativeTensor[i] - 1 end

			end
			
		else
			
			if (topPositionIndex == labelTensor) then partialFirstDerivativeTensor = partialFirstDerivativeTensor - 1 end
			
		end
		
	end
	
	return partialFirstDerivativeTensor
	
end

local function calculateSoftmaxSparseCategoricalCrossEntropyFirstDerivativeTensor(partialFirstDerivativeTensor, labelTensor, dimension, dimensionSizeArray, numberOfDimensions, currentDimension)
	
	local currentDimensionSize = dimensionSizeArray[currentDimension]

	if (currentDimension == dimension) then
		
		local subLabelTensor = labelTensor[1]
		
		local subDimensionSizeArray = {}
		
		for i = (currentDimension + 1), numberOfDimensions, 1 do table.insert(subDimensionSizeArray, dimensionSizeArray[i]) end
		
		local subNumberOfDimensions = #subDimensionSizeArray
		
		for i = 1, currentDimensionSize, 1 do
			
			partialFirstDerivativeTensor[i] = subtractMinusOneAtTopPositionIndexFromTensor(partialFirstDerivativeTensor[i], subLabelTensor, subDimensionSizeArray, subNumberOfDimensions, 1, i)

		end
		
	else
		
		local nextDimension = currentDimension + 1
		
		for i = 1, currentDimensionSize, 1 do

			calculateSoftmaxSparseCategoricalCrossEntropyFirstDerivativeTensor(partialFirstDerivativeTensor[i], labelTensor[i], dimension, dimensionSizeArray, numberOfDimensions, nextDimension)

		end

	end	
	
end

function FusedCostFunctions.SigmoidBinaryCrossEntropy(parameterDictionary)

	local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]

	local inputTensorArray = {inputTensor, labelTensor}

	local sigmoidFunction = function (z) return (1 / (1 + math.exp(-z))) end

	local binaryCrossEntropyFunction = function (labelValue, generatedLabelValue) return -(labelValue * math.log(generatedLabelValue) + (1 - labelValue) * math.log(1 - generatedLabelValue)) end

	local pureInputTensor = AutomaticDifferentiationTensor:fetchValue{inputTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

	local sigmoidTensor = AqwamTensorLibrary:applyFunction(sigmoidFunction, inputTensor)

	local binaryCrossEntropyTensor = AqwamTensorLibrary:applyFunction(binaryCrossEntropyFunction, pureLabelTensor, sigmoidTensor)

	local sumBinaryCrossEntropyValue = AqwamTensorLibrary:sum(binaryCrossEntropyTensor)

	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = sumBinaryCrossEntropyValue / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local inputTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureInputTensor = AutomaticDifferentiationTensor:fetchValue{inputTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		local sigmoidTensor = AqwamTensorLibrary:applyFunction(sigmoidFunction, pureInputTensor)

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{inputTensor}) then

			if (inputTensor:getIsFirstDerivativeTensorRequired()) then

				local partialFirstDerivativeTensor = AqwamTensorLibrary:subtract(sigmoidTensor, pureLabelTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureInputTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				inputTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

			if (labelTensor:getIsFirstDerivativeTensorRequired()) then

				local functionToApply = function (z) return (math.log(1 - z) - math.log(z)) end

				local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, sigmoidTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedNegativeFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedNegativeFirstDerivativeTensor}

			end

		end

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

function FusedCostFunctions.SoftmaxCategoricalCrossEntropy(parameterDictionary)

	local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]

	local dimension = parameterDictionary.dimension or parameterDictionary[3] or 1

	local inputTensorArray = {inputTensor, labelTensor}

	local functionToApply = function (labelValue, generatedLabelValue) return (labelValue * math.log(generatedLabelValue)) end

	local pureinputTensor = AutomaticDifferentiationTensor:fetchValue{inputTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

	local exponentTensor = AqwamTensorLibrary:applyFunction(math.exp, pureinputTensor)

	local sumExponentTensor = AqwamTensorLibrary:sum(exponentTensor, dimension)

	local softmaxTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentTensor)

	local categoricalCrossEntropyTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureLabelTensor, softmaxTensor)

	local sumCategoricalCrossEntropyValue = AqwamTensorLibrary:sum(categoricalCrossEntropyTensor)

	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = -sumCategoricalCrossEntropyValue / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local inputTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureinputTensor = AutomaticDifferentiationTensor:fetchValue{inputTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		local exponentTensor = AqwamTensorLibrary:applyFunction(math.exp, pureinputTensor)

		local sumExponentTensor = AqwamTensorLibrary:sum(exponentTensor, dimension)

		local softmaxTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentTensor)

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{inputTensor}) then

			if (inputTensor:getIsFirstDerivativeTensorRequired()) then

				local partialFirstDerivativeTensor = AqwamTensorLibrary:subtract(softmaxTensor, pureLabelTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureinputTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				inputTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

			if (labelTensor:getIsFirstDerivativeTensorRequired()) then

				local partialFirstDerivativeTensor = AqwamTensorLibrary:logarithm(softmaxTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, -numberOfData)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

function FusedCostFunctions.StableSoftmaxCategoricalCrossEntropy(parameterDictionary)

	local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]

	local dimension = parameterDictionary.dimension or parameterDictionary[3] or 1

	local inputTensorArray = {inputTensor, labelTensor}

	local functionToApply = function (labelValue, generatedLabelValue) return (labelValue * math.log(generatedLabelValue)) end

	local pureinputTensor = AutomaticDifferentiationTensor:fetchValue{inputTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

	local maximumValue = AqwamTensorLibrary:findMaximumValue(pureinputTensor)

	local subtractedZTensor =  AqwamTensorLibrary:subtract(pureinputTensor, maximumValue)

	local exponentTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

	local sumExponentTensor = AqwamTensorLibrary:sum(exponentTensor, dimension)

	local softmaxTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentTensor)

	local categoricalCrossEntropyTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureLabelTensor, softmaxTensor)

	local sumCategoricalCrossEntropyValue = AqwamTensorLibrary:sum(categoricalCrossEntropyTensor)

	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = -sumCategoricalCrossEntropyValue / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local inputTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureinputTensor = AutomaticDifferentiationTensor:fetchValue{inputTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		local maximumValue = AqwamTensorLibrary:findMaximumValue(pureinputTensor)

		local subtractedZTensor =  AqwamTensorLibrary:subtract(pureinputTensor, maximumValue)

		local exponentTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

		local sumExponentTensor = AqwamTensorLibrary:sum(exponentTensor, dimension)

		local softmaxTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentTensor)

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{inputTensor}) then

			if (inputTensor:getIsFirstDerivativeTensorRequired()) then

				local partialFirstDerivativeTensor = AqwamTensorLibrary:subtract(softmaxTensor, pureLabelTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureinputTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				inputTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

			if (labelTensor:getIsFirstDerivativeTensorRequired()) then

				local partialFirstDerivativeTensor = AqwamTensorLibrary:logarithm(softmaxTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, -numberOfData)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

function FusedCostFunctions.SoftmaxSparseCategoricalCrossEntropy(parameterDictionary)

	local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]

	local dimension = parameterDictionary.dimension or parameterDictionary[3] or 1

	local inputTensorArray = {inputTensor, labelTensor}

	local functionToApply = function (labelValue, generatedLabelValue) return (labelValue * math.log(generatedLabelValue)) end

	local pureinputTensor = AutomaticDifferentiationTensor:fetchValue{inputTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

	local exponentTensor = AqwamTensorLibrary:applyFunction(math.exp, pureinputTensor)

	local sumExponentTensor = AqwamTensorLibrary:sum(exponentTensor, dimension)

	local softmaxTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentTensor)

	local categoricalCrossEntropyTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureLabelTensor, softmaxTensor)

	local sumCategoricalCrossEntropyValue = AqwamTensorLibrary:sum(categoricalCrossEntropyTensor)

	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = -sumCategoricalCrossEntropyValue / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local inputTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureinputTensor = AutomaticDifferentiationTensor:fetchValue{inputTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		local exponentTensor = AqwamTensorLibrary:applyFunction(math.exp, pureinputTensor)

		local sumExponentTensor = AqwamTensorLibrary:sum(exponentTensor, dimension)

		local softmaxTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentTensor)

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{inputTensor}) then

			if (inputTensor:getIsFirstDerivativeTensorRequired()) then
				
				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureinputTensor)
				
				local partialFirstDerivativeTensor = AqwamTensorLibrary:copy(softmaxTensor)
				
				calculateSoftmaxSparseCategoricalCrossEntropyFirstDerivativeTensor(partialFirstDerivativeTensor, pureLabelTensor, dimension, dimensionSizeArray, #dimensionSizeArray, 1)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				inputTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

			if (labelTensor:getIsFirstDerivativeTensorRequired()) then

				local partialFirstDerivativeTensor = AqwamTensorLibrary:logarithm(softmaxTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, -numberOfData)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

function FusedCostFunctions.StableSoftmaxSparseCategoricalCrossEntropy(parameterDictionary)

	local inputTensor = parameterDictionary.inputTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]

	local dimension = parameterDictionary.dimension or parameterDictionary[3] or 1

	local inputTensorArray = {inputTensor, labelTensor}

	local functionToApply = function (labelValue, generatedLabelValue) return (labelValue * math.log(generatedLabelValue)) end

	local pureinputTensor = AutomaticDifferentiationTensor:fetchValue{inputTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

	local maximumValue = AqwamTensorLibrary:findMaximumValue(pureinputTensor)

	local subtractedZTensor =  AqwamTensorLibrary:subtract(pureinputTensor, maximumValue)

	local exponentTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

	local sumExponentTensor = AqwamTensorLibrary:sum(exponentTensor, dimension)

	local softmaxTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentTensor)

	local categoricalCrossEntropyTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureLabelTensor, softmaxTensor)

	local sumCategoricalCrossEntropyValue = AqwamTensorLibrary:sum(categoricalCrossEntropyTensor)

	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = -sumCategoricalCrossEntropyValue / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local inputTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureinputTensor = AutomaticDifferentiationTensor:fetchValue{inputTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		local maximumValue = AqwamTensorLibrary:findMaximumValue(pureinputTensor)

		local subtractedZTensor =  AqwamTensorLibrary:subtract(pureinputTensor, maximumValue)

		local exponentTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

		local sumExponentTensor = AqwamTensorLibrary:sum(exponentTensor, dimension)

		local softmaxTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentTensor)

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{inputTensor}) then

			if (inputTensor:getIsFirstDerivativeTensorRequired()) then

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureinputTensor)

				local partialFirstDerivativeTensor = AqwamTensorLibrary:copy(softmaxTensor)

				calculateSoftmaxSparseCategoricalCrossEntropyFirstDerivativeTensor(partialFirstDerivativeTensor, pureLabelTensor, dimension, dimensionSizeArray, #dimensionSizeArray, 1)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				inputTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

			if (labelTensor:getIsFirstDerivativeTensorRequired()) then

				local partialFirstDerivativeTensor = AqwamTensorLibrary:logarithm(softmaxTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, -numberOfData)
				
				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

return FusedCostFunctions
