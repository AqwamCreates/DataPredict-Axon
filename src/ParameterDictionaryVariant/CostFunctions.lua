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
	
	local epsilon = parameterDictionary.epsilon or parameterDictionary[3]
	
	local inputTensorArray = {generatedLabelTensor, labelTensor}
	
	local functionToApply = function (generatedLabelValue, labelValue) return -(labelValue * math.log(generatedLabelValue) + (1 - labelValue) * math.log(1 - generatedLabelValue)) end
	
	local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

	local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

	local binaryCrossEntropyTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor)
	
	local sumBinaryCrossEntropyValue = AqwamTensorLibrary:sum(binaryCrossEntropyTensor)
	
	local numberOfData = getNumberOfData(labelTensor)

	local resultValue = sumBinaryCrossEntropyValue / numberOfData
	
	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local generatedLabelTensor = inputTensorArray[1]

		local labelTensor = inputTensorArray[2]

		local pureGeneratedLabelTensor = AutomaticDifferentiationTensor:fetchValue{generatedLabelTensor}

		local pureLabelTensor = AutomaticDifferentiationTensor:fetchValue{labelTensor}

		local lossTensor = AqwamTensorLibrary:subtract(pureGeneratedLabelTensor, pureLabelTensor)

		local scale = 1 / numberOfData

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
		
		local functionToApply = function (predictedValue, labelValue) 

			local isLabelValueEqualTo1 = (labelValue == 1)

			local pT = (isLabelValueEqualTo1 and predictedValue) or (1 - predictedValue)

			local focalLossValue = -alpha * ((1 - pT) ^ gamma) * ((gamma * pT * math.log(pT)) + pT - 1)

			return focalLossValue

		end

		local lossTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureGeneratedLabelTensor, pureLabelTensor)

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{generatedLabelTensor}) then

			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureGeneratedLabelTensor)

			local collapsedLossTensor = collapseTensor(lossTensor, dimensionSizeArray)

			generatedLabelTensor:differentiate{collapsedLossTensor}

		end

		if (AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{labelTensor}) then

			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureLabelTensor)

			local collapsedLossTensor = collapseTensor(lossTensor, dimensionSizeArray)

			labelTensor:differentiate{collapsedLossTensor}

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

		local scale = 1 / numberOfData
		
		local functionToApply = function(x)
			if x > 0 then return 1
			elseif x < 0 then return -1
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
