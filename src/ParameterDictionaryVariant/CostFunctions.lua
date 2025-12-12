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

function CostFunctions.FastEpsilonInsentitiveLoss(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]

	local epsilon = parameterDictionary.epsilon or parameterDictionary[3] or 1

	local cValue = parameterDictionary.cValue or parameterDictionary[4] or 1
	
	local inputTensorArray = {generatedLabelTensor, labelTensor}
	
	local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

	local errorTensor = AqwamTensorLibrary:subtract(pureGeneratedLabelTensor, pureLabelTensor)

	local epsilonErrorTensor = AqwamTensorLibrary:subtract(errorTensor, epsilon)

	local positiveSlackVariableTensor = AqwamTensorLibrary:applyFunction(math.max, 0, epsilonErrorTensor)

	local negativeSlackVariableTensor = AqwamTensorLibrary:applyFunction(math.max, 0, -epsilonErrorTensor)

	local slackVariableTensor = AqwamTensorLibrary:add(positiveSlackVariableTensor, negativeSlackVariableTensor)

	local sumSlackVariableValue = AqwamTensorLibrary:sum(slackVariableTensor)

	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = (cValue * sumSlackVariableValue) / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local generatedLabelTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		local functionToApply = function(errorValue) return ((errorValue > epsilon) and (errorValue - epsilon)) or ((errorValue < -epsilon) and (errorValue + epsilon)) or 0 end

		local partialFirstDerivativeTensorPart1 = AqwamTensorLibrary:applyFunction(functionToApply, errorTensor)
		
		partialFirstDerivativeTensorPart1 = AqwamTensorLibrary:multiply(cValue, partialFirstDerivativeTensorPart1)
		
		partialFirstDerivativeTensorPart1 = AqwamTensorLibrary:divide(partialFirstDerivativeTensorPart1, numberOfData)

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then

			if (generatedLabelTensor:getIsFirstDerivativeTensorRequired()) then

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensorPart1)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

			if (labelTensor:getIsFirstDerivativeTensorRequired()) then

				local negativeFirstDerivativeTensor = AqwamTensorLibrary:multiply(-1, firstDerivativeTensor, partialFirstDerivativeTensorPart1)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedNegativeFirstDerivativeTensor = collapseTensor(negativeFirstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedNegativeFirstDerivativeTensor}

			end

		end

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

function CostFunctions.FastHingeLoss(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local inputTensorArray = {generatedLabelTensor, labelTensor}
	
	local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}
	
	local hingeLossTensorPart1 = AqwamTensorLibrary:multiply(pureLabelTensor, pureGeneratedLabelTensor)

	local hingeLossTensorPart2 = AqwamTensorLibrary:subtract(1, hingeLossTensorPart1)

	local hingeLossTensor = AqwamTensorLibrary:applyFunction(math.max, 0, hingeLossTensorPart1)

	local sumHingeLossTensorValue = AqwamTensorLibrary:sum(hingeLossTensor)

	local numberOfData = getNumberOfData(labelTensor)
	
	local resultValue = sumHingeLossTensorValue / numberOfData

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local generatedLabelTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		local indicatorFunction = function(x) return ((x > 0) and 1) or 0 end

		local indicatorTensor = AqwamTensorLibrary:applyFunction(indicatorFunction, hingeLossTensorPart2)
		
		local partialFirstDerivativeTensorPart1 = AqwamTensorLibrary:divide(indicatorTensor, -numberOfData)

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then

			if (generatedLabelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local partialFirstDerivativeTensor = AqwamTensorLibrary:multiply(partialFirstDerivativeTensorPart1, pureLabelTensor) 

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

			if (labelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local partialFirstDerivativeTensor = AqwamTensorLibrary:multiply(partialFirstDerivativeTensorPart1, labelTensor) 

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedFirstDerivativeTensor}

			end

		end

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

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
	
	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local generatedLabelTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}
		
		local subtractedPureGeneratedLabelTensor = AqwamTensorLibrary:subtract(1, pureGeneratedLabelTensor)

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then
			
			if (generatedLabelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local partialFirstDerivativeTensorPart1 = AqwamTensorLibrary:subtract(pureLabelTensor, pureGeneratedLabelTensor)

				local partialFirstDerivativeTensorPart2 = AqwamTensorLibrary:multiply(subtractedPureGeneratedLabelTensor, pureGeneratedLabelTensor)

				local partialFirstDerivativeTensor = AqwamTensorLibrary:divide(partialFirstDerivativeTensorPart1, partialFirstDerivativeTensorPart2)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)
				
				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}
				
			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then
			
			if (labelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local partialFirstDerivativeTensorPart1 = AqwamTensorLibrary:logarithm(subtractedPureGeneratedLabelTensor)

				local partialFirstDerivativeTensorPart2 = AqwamTensorLibrary:logarithm(pureGeneratedLabelTensor)

				local partialFirstDerivativeTensor = AqwamTensorLibrary:subtract(partialFirstDerivativeTensorPart1, partialFirstDerivativeTensorPart2)

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

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local generatedLabelTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then
			
			if (generatedLabelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local partialFirstDerivativeTensor = AqwamTensorLibrary:divide(pureLabelTensor, pureGeneratedLabelTensor)

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)
				
				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, -numberOfData)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}
				
			end
			
		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then
			
			if (labelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local partialFirstDerivativeTensor = AqwamTensorLibrary:logarithm(pureGeneratedLabelTensor)

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
	
	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)
		
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

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then
			
			if (generatedLabelTensor:getIsFirstDerivativeTensorRequired()) then
				
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
				
				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedLossTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedLossTensor}
				
			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then
			
			if (labelTensor:getIsFirstDerivativeTensorRequired()) then
				
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

				local firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, partialFirstDerivativeTensor)
				
				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedLossTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedLossTensor}
				
			end

		end

	end
	
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

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local generatedLabelTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		local lossTensor = AqwamTensorLibrary:subtract(pureGeneratedLabelTensor, pureLabelTensor)
		
		lossTensor = AqwamTensorLibrary:applyFunction(math.sign, lossTensor)

		firstDerivativeTensor = AqwamTensorLibrary:multiply(lossTensor, firstDerivativeTensor)
		
		firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then
			
			if (generatedLabelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}
				
			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then
			
			if (labelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local negativeFirstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, -1)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedNegativeFirstDerivativeTensor = collapseTensor(negativeFirstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedNegativeFirstDerivativeTensor}
				
			end

		end

	end

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

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local generatedLabelTensor = inputTensorArray[1]
		
		local labelTensor = inputTensorArray[2]
		
		local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		local lossTensor = AqwamTensorLibrary:subtract(pureGeneratedLabelTensor, pureLabelTensor)
		
		firstDerivativeTensor = AqwamTensorLibrary:multiply(lossTensor, firstDerivativeTensor, 2)
		
		firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, numberOfData)

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then
			
			if (generatedLabelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

				local collapsedFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				generatedLabelTensor:differentiate{collapsedFirstDerivativeTensor}
				
			end

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then
			
			if (labelTensor:getIsFirstDerivativeTensorRequired()) then
				
				local negativeFirstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, -1)

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

				local collapsedNegativeFirstDerivativeTensor = collapseTensor(negativeFirstDerivativeTensor, dimensionSizeArray)

				labelTensor:differentiate{collapsedNegativeFirstDerivativeTensor}
				
			end

		end

	end

	return AutomaticDifferentiationTensor.new({resultValue, PartialFirstDerivativeFunction, inputTensorArray})

end

------------------------------------------

function CostFunctions.EpsilonInsentitiveLoss(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local epsilon = parameterDictionary.epsilon or parameterDictionary[3] or 1
	
	local cValue = parameterDictionary.cValue or parameterDictionary[4] or 1
	
	local errorTensor = generatedLabelTensor - labelTensor
	
	local epsilonErrorTensor = errorTensor - epsilon
	
	local positiveSlackVariableTensor = AutomaticDifferentiationTensor.maximum{0, epsilonErrorTensor}
	
	local negativeSlackVariableTensor = AutomaticDifferentiationTensor.maximum{0, -epsilonErrorTensor}
	
	local slackVariableTensor = positiveSlackVariableTensor + negativeSlackVariableTensor
	
	local sumSlackVariableValue = slackVariableTensor:sum()

	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = (cValue * sumSlackVariableValue) / numberOfData

	return resultValue

end

function CostFunctions.HingeLoss(parameterDictionary)

	local generatedLabelTensor = parameterDictionary.generatedLabelTensor or parameterDictionary[1]

	local labelTensor = parameterDictionary.labelTensor or parameterDictionary[2]
	
	local hingeLossTensorPart1 = 1 - (generatedLabelTensor * labelTensor)

	local hingeLossTensor = AutomaticDifferentiationTensor.maximum{0, hingeLossTensorPart1}

	local sumHingeLossTensorTensor = hingeLossTensor:sum()

	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = sumHingeLossTensorTensor / numberOfData

	return resultValue

end

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
