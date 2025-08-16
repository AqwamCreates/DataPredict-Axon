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

local CostFunctions = {}

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

function CostFunctions.FastBinaryCrossEntropy(parameterDictionary)
	
	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]
	
	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local inputTensorArray = {generatedLabelTensor, labelTensor}
	
	local functionToApply = function (labelValue, generatedLabelValue) return -(labelValue * math.log(generatedLabelValue) + (1 - labelValue) * math.log(1 - generatedLabelValue)) end
	
	local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

	local binaryCrossEntropyTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureLabelTensor, pureGeneratedLabelTensor)
	
	local sumBinaryCrossEntropyValue = AqwamTensorLibrary:sum(binaryCrossEntropyTensor)
	
	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = sumBinaryCrossEntropyValue / numberOfData
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local generatedLabelTensor = inputTensorArray[1]

			local labelTensor = inputTensorArray[2]

			local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

			local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

			local scale = 1 / numberOfData

			local subtractedPureGeneratedLabelTensor = AqwamTensorLibrary:subtract(1, pureGeneratedLabelTensor)

			if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then

				local partialFirstDerivativeTensorPart1 = AqwamTensorLibrary:subtract(pureLabelTensor, pureGeneratedLabelTensor)

				local partialFirstDerivativeTensorPart2 = AqwamTensorLibrary:multiply(subtractedPureGeneratedLabelTensor, pureGeneratedLabelTensor)

				local partialFirstDerivativeTensor = AqwamTensorLibrary:divide(partialFirstDerivativeTensorPart1, partialFirstDerivativeTensorPart2)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor, scale)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

			if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

				local partialFirstDerivativeTensorPart1 = AqwamTensorLibrary:logarithm(subtractedPureGeneratedLabelTensor)

				local partialFirstDerivativeTensorPart2 = AqwamTensorLibrary:logarithm(pureGeneratedLabelTensor)

				local partialFirstDerivativeTensor = AqwamTensorLibrary:subtract(partialFirstDerivativeTensorPart1, partialFirstDerivativeTensorPart2)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor, scale)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedNegativeFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedNegativeFirstDerivativeTensor}

			end

		end
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end
	
	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})
	
end

function CostFunctions.FastCategoricalCrossEntropy(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local inputTensorArray = {generatedLabelTensor, labelTensor}
	
	local functionToApply = function (labelValue, generatedLabelValue) return (labelValue * math.log(generatedLabelValue)) end
	
	local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}
	
	local categoricalCrossEntropyTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureLabelTensor, pureGeneratedLabelTensor)

	local sumCategoricalCrossEntropyValue = AqwamTensorLibrary:sum(categoricalCrossEntropyTensor)
	
	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = -sumCategoricalCrossEntropyValue / numberOfData
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local generatedLabelTensor = inputTensorArray[1]

			local labelTensor = inputTensorArray[2]

			local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

			local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

			local scale = -1 / numberOfData

			if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then

				local partialFirstDerivativeTensor = AqwamTensorLibrary:divide(pureLabelTensor, pureGeneratedLabelTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor, scale)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

			if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

				local partialFirstDerivativeTensor = AqwamTensorLibrary:logarithm(pureGeneratedLabelTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor, scale)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})
	
end

function CostFunctions.FastFocalLoss(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local alpha = parameterDictionary.alpha or parameterDictionary[3] or defaultAlpha
	
	local gamma = parameterDictionary.gamma or parameterDictionary[4] or defaultGamma
	
	local inputTensorArray = {generatedLabelTensor, labelTensor}
	
	local functionToApply = function (generatedLabelValue, labelValue) 

		local isLabelValueEqualTo1 = (labelValue == 1)

		local pT = (isLabelValueEqualTo1 and generatedLabelValue) or (1 - generatedLabelValue)

		local focalLossValue = -alpha * ((1 - pT) ^ gamma) * math.log(pT)

		return focalLossValue

	end
	
	local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

	local focalLossTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor)

	local sumFocalLossValue = AqwamTensorLibrary:sum(focalLossTensor)
	
	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = sumFocalLossValue / numberOfData
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local generatedLabelTensor = inputTensorArray[1]

			local labelTensor = inputTensorArray[2]

			local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

			local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

			local pureNegativeGeneratedLabelTensor = AqwamTensorLibrary:unaryMinus(pureGeneratedLabelTensor)

			local pureNegativeLabelTensor = AqwamTensorLibrary:unaryMinus(pureLabelTensor)

			local functionToApply = function (labelValue, generatedValue) 

				local pT = (labelValue * generatedValue) + (1 - labelValue) * (1 - generatedValue)

				local focalLossValue = -alpha * ((1 - pT) ^ gamma) * ((gamma * pT * math.log(pT)) + pT - 1)

				return focalLossValue

			end

			local scale = 1 / numberOfData

			if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then

				local twoABTensor = AqwamTensorLibrary:multiply(2, AqwamTensorLibrary:multiply(pureLabelTensor, pureGeneratedLabelTensor))

				local zTensor = AqwamTensorLibrary:add(twoABTensor, AqwamTensorLibrary:add(pureNegativeLabelTensor, AqwamTensorLibrary:add(pureNegativeGeneratedLabelTensor, 1)))

				local aPlusBTensor = AqwamTensorLibrary:add(pureLabelTensor, pureGeneratedLabelTensor)

				local aPlusBMinusTwoTensor = AqwamTensorLibrary:add(aPlusBTensor, -2)

				local minusGeneratedLabelTensor = AqwamTensorLibrary:multiply(-1, pureGeneratedLabelTensor)

				local basePowerTensor = AqwamTensorLibrary:multiply(minusGeneratedLabelTensor, aPlusBMinusTwoTensor)

				local powerTermTensor = AqwamTensorLibrary:power(basePowerTensor, gamma)

				local oneMinusTwoATensor = AqwamTensorLibrary:add(1, AqwamTensorLibrary:multiply(-2, pureLabelTensor))

				local logZTensor = AqwamTensorLibrary:logarithm(zTensor)

				local twoGeneratedLabelTensor = AqwamTensorLibrary:multiply(2, pureGeneratedLabelTensor)

				local aPlusTwoBMinusTwoTensor = AqwamTensorLibrary:add(pureLabelTensor, AqwamTensorLibrary:add(twoGeneratedLabelTensor, -2))

				local numeratorTermTwoTensor = AqwamTensorLibrary:multiply(gamma, AqwamTensorLibrary:multiply(aPlusTwoBMinusTwoTensor, logZTensor))

				local denominatorTermTwoTensor = AqwamTensorLibrary:multiply(pureGeneratedLabelTensor, aPlusBMinusTwoTensor)

				local termOneTensor = AqwamTensorLibrary:divide(oneMinusTwoATensor, zTensor)

				local termTwoTensor = AqwamTensorLibrary:divide(numeratorTermTwoTensor, denominatorTermTwoTensor)

				local bracketExpressionTensor = AqwamTensorLibrary:subtract(termOneTensor, termTwoTensor)

				local partialFirstDerivativeTensor = AqwamTensorLibrary:multiply(alpha, AqwamTensorLibrary:multiply(powerTermTensor, bracketExpressionTensor))

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedLossTensor = collapseTensor(partialFirstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedLossTensor}

			end

			if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

				local twoABTensor = AqwamTensorLibrary:multiply(2, AqwamTensorLibrary:multiply(pureLabelTensor, pureGeneratedLabelTensor))

				local zTensor = AqwamTensorLibrary:add(twoABTensor, AqwamTensorLibrary:add(pureNegativeLabelTensor, AqwamTensorLibrary:add(pureNegativeGeneratedLabelTensor, 1)))

				local aPlusBTensor = AqwamTensorLibrary:add(pureLabelTensor, pureGeneratedLabelTensor)

				local aPlusBMinusTwoTensor = AqwamTensorLibrary:add(aPlusBTensor, -2)

				local minusGeneratedLabelTensor = AqwamTensorLibrary:multiply(-1, pureGeneratedLabelTensor)

				local basePowerTensor = AqwamTensorLibrary:multiply(minusGeneratedLabelTensor, aPlusBMinusTwoTensor)

				local powerTermTensor = AqwamTensorLibrary:power(basePowerTensor, gamma)

				local oneMinusTwoBTensor = AqwamTensorLibrary:add(1, AqwamTensorLibrary:multiply(-2, pureGeneratedLabelTensor))

				local logZTensor = AqwamTensorLibrary:logarithm(zTensor)

				local termOneTensor = AqwamTensorLibrary:divide(oneMinusTwoBTensor, zTensor)

				local gammaLogZTensor = AqwamTensorLibrary:multiply(gamma, logZTensor)

				local termTwoTensor = AqwamTensorLibrary:divide(gammaLogZTensor, aPlusBMinusTwoTensor)

				local bracketExpressionTensor = AqwamTensorLibrary:subtract(termOneTensor, termTwoTensor)

				local partialFirstDerivativeTensor = AqwamTensorLibrary:multiply(alpha, AqwamTensorLibrary:multiply(powerTermTensor, bracketExpressionTensor))

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor, scale)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedLossTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedLossTensor}

			end

		end
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end
	
	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})
	
end

function CostFunctions.FastMeanAbsoluteError(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local inputTensorArray = {generatedLabelTensor, labelTensor}

	local functionToApply = function (generatedLabelValue, labelValue) return math.abs(generatedLabelValue - labelValue) end
	
	local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

	local absoluteErrorTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor)

	local sumAbsoluteErrorValue = AqwamTensorLibrary:sum(absoluteErrorTensor)
	
	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = sumAbsoluteErrorValue / numberOfData
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local generatedLabelTensor = inputTensorArray[1]

			local labelTensor = inputTensorArray[2]

			local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

			local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

			local lossTensor = AqwamTensorLibrary:subtract(pureGeneratedLabelTensor, pureLabelTensor)

			local scale = 1 / numberOfData

			local functionToApply = function(x)

				if (x > 0) then return 1

				elseif (x < 0) then return -1

				else return 0 end

			end

			lossTensor = AqwamTensorLibrary:applyFunction(functionToApply, lossTensor)

			firstDerivativeTensor = AqwamTensorLibrary:multiply(lossTensor, firstDerivativeTensor, scale)

			if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

			if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

				local negativeFirstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, -1)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedNegativeFirstDerivativeTensor = collapseTensor(negativeFirstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedNegativeFirstDerivativeTensor}

			end

		end
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

function CostFunctions.FastMeanSquaredError(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local inputTensorArray = {generatedLabelTensor, labelTensor}

	local functionToApply = function (generatedLabelValue, labelValue) return math.pow((generatedLabelValue - labelValue), 2) end

	local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

	local squaredErrorTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor)

	local sumSquaredErrorValue = AqwamTensorLibrary:sum(squaredErrorTensor)
	
	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = sumSquaredErrorValue / numberOfData
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local generatedLabelTensor = inputTensorArray[1]

			local labelTensor = inputTensorArray[2]

			local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

			local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

			local lossTensor = AqwamTensorLibrary:subtract(pureGeneratedLabelTensor, pureLabelTensor)

			local scale = 2 / numberOfData

			firstDerivativeTensor = AqwamTensorLibrary:multiply(lossTensor, firstDerivativeTensor, scale)

			if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

			if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

				local negativeFirstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, -1)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedNegativeFirstDerivativeTensor = collapseTensor(negativeFirstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedNegativeFirstDerivativeTensor}

			end

		end
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

------------------------------------------

function CostFunctions.BinaryCrossEntropy(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local binaryCrossEntropyTensor = -(labelTensor * AutomaticDifferentiationTensor.logarithm{generatedLabelTensor} + (1 - labelTensor) * AutomaticDifferentiationTensor.logarithm{1 - generatedLabelTensor})

	local sumBinaryCrossEntropyTensor = binaryCrossEntropyTensor:sum()
	
	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = sumBinaryCrossEntropyTensor / numberOfData

	return resultValue

end

function CostFunctions.CategoricalCrossEntropy(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local categoricalCrossEntropyTensor = labelTensor * AutomaticDifferentiationTensor.logarithm{generatedLabelTensor}

	local sumCategoricalCrossEntropy = categoricalCrossEntropyTensor:sum()
	
	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = -sumCategoricalCrossEntropy / numberOfData
	
	return resultValue

end

function CostFunctions.FocalLoss(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]

	local alpha = parameterDictionary.alpha or parameterDictionary[3] or defaultAlpha

	local gamma = parameterDictionary.gamma or parameterDictionary[4] or defaultGamma
	
	local focalLossTensorPart1 = (labelTensor * generatedLabelTensor) + (1 - labelTensor) * (1 - generatedLabelTensor)
	
	local focalLossTensor =  -alpha * ((1 - focalLossTensorPart1) ^ gamma) * AutomaticDifferentiationTensor.logarithm(focalLossTensorPart1)

	local sumFocalLossValue = focalLossTensor:sum()
	
	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = sumFocalLossValue / numberOfData

	return resultValue

end

function CostFunctions.MeanAbsoluteError(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local absoluteErrorTensor = (generatedLabelTensor - labelTensor):absolute()

	local sumAbsoluteError = absoluteErrorTensor:sum()
	
	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = sumAbsoluteError / numberOfData

	return resultValue

end

function CostFunctions.MeanSquaredError(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local squaredErrorTensor = (generatedLabelTensor - labelTensor)^2

	local sumSquaredErrorValue = squaredErrorTensor:sum()
	
	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = sumSquaredErrorValue / numberOfData

	return resultValue

end

return CostFunctions
