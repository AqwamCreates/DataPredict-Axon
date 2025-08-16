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

local DropoutLayers = {}

local defaultDropoutRate = 0.5

function DropoutLayers.Dropout(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local dropoutRate = parameterDictionary.dropoutRate or parameterDictionary[2]

	local inputTensorArray = {tensor}

	local nonDropoutRate = 1 - dropoutRate

	local scalingFactor = 1 / nonDropoutRate

	local functionToApply = function (x)

		local isDroppedOut = (math.random() > nonDropoutRate)

		return (isDroppedOut and 0) or (x * scalingFactor)

	end
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureTensor)

	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local tensor = inputTensorArray[1]

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

			tensor:differentiate{firstDerivativeTensor}

		end

	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function DropoutLayers.Dropout1D(parameterDictionary)
	
	local tensor = parameterDictionary.tensor or parameterDictionary[1]
	
	local dropoutRate = parameterDictionary.dropoutRate or parameterDictionary[2]
	
	local inputTensorArray = {tensor}
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}
	
	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureTensor)

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	if (tensorNumberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial dropout function. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. tensorNumberOfDimensions .. " dimensions.") end

	local numberOfData = tensorDimensionSizeArray[1]

	local dimensionSizeAtSecondDimension = tensorDimensionSizeArray[2]

	local dimensionSizeAtThirdDimension = tensorDimensionSizeArray[3]

	local nonDropoutRate = 1 - dropoutRate

	local scalingFactor = 1 / nonDropoutRate

	local resultTensor = AqwamTensorLibrary:copy(pureTensor)

	for i = 1, numberOfData, 1 do

		for j = 1, dimensionSizeAtSecondDimension, 1 do

			if (math.random() > nonDropoutRate) then resultTensor[i][j] = table.create(dimensionSizeAtThirdDimension, 0) end

		end

	end

	resultTensor = AqwamTensorLibrary:multiply(resultTensor, scalingFactor)
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local tensor = inputTensorArray[1]

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

			tensor:differentiate{firstDerivativeTensor}

		end

	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end
	
	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})
	
end

function DropoutLayers.Dropout2D(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local dropoutRate = parameterDictionary.dropoutRate or parameterDictionary[2]

	local inputTensorArray = {tensor}
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	if (tensorNumberOfDimensions ~= 4) then error("Unable to pass the input tensor to the 2D spatial dropout function. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. tensorNumberOfDimensions .. " dimensions.") end

	local numberOfData = tensorDimensionSizeArray[1]

	local dimensionSizeAtSecondDimension = tensorDimensionSizeArray[2]

	local dropoutTensorDimensionSizeArray = {tensorDimensionSizeArray[3], tensorDimensionSizeArray[4]}

	local nonDropoutRate = 1 - dropoutRate

	local scalingFactor = 1 / nonDropoutRate

	local resultTensor = AqwamTensorLibrary:copy(pureTensor)

	for i = 1, numberOfData, 1 do

		for j = 1, dimensionSizeAtSecondDimension, 1 do

			if (math.random() > nonDropoutRate) then resultTensor[i][j] = AqwamTensorLibrary:createTensor(dropoutTensorDimensionSizeArray, 0) end

		end

	end

	resultTensor = AqwamTensorLibrary:multiply(resultTensor, scalingFactor)

	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local tensor = inputTensorArray[1]

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

			tensor:differentiate{firstDerivativeTensor}

		end

	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function DropoutLayers.Dropout3D(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local dropoutRate = parameterDictionary.dropoutRate or parameterDictionary[2]

	local inputTensorArray = {tensor}
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	if (tensorNumberOfDimensions ~= 5) then error("Unable to pass the input tensor to the 3D spatial dropout function. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. tensorNumberOfDimensions .. " dimensions.") end

	local numberOfData = tensorDimensionSizeArray[1]

	local dimensionSizeAtSecondDimension = tensorDimensionSizeArray[2]

	local dropoutTensorDimensionSizeArray = {tensorDimensionSizeArray[3], tensorDimensionSizeArray[4], tensorDimensionSizeArray[5]}

	local nonDropoutRate = 1 - dropoutRate

	local scalingFactor = 1 / nonDropoutRate

	local resultTensor = AqwamTensorLibrary:copy(pureTensor)

	for i = 1, numberOfData, 1 do

		for j = 1, dimensionSizeAtSecondDimension, 1 do

			if (math.random() > nonDropoutRate) then resultTensor[i][j] = AqwamTensorLibrary:createTensor(dropoutTensorDimensionSizeArray, 0) end

		end

	end

	resultTensor = AqwamTensorLibrary:multiply(resultTensor, scalingFactor)

	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local tensor = inputTensorArray[1]

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

			tensor:differentiate{firstDerivativeTensor}

		end

	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function DropoutLayers.DropoutND(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local dropoutRate = parameterDictionary.dropoutRate or parameterDictionary[2]

	local inputTensorArray = {tensor}
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureTensor)

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local numberOfData = tensorDimensionSizeArray[1]

	local dimensionSizeAtSecondDimension = tensorDimensionSizeArray[2]

	local dropoutTensorDimensionSizeArray = {}
	
	for i = 3, tensorNumberOfDimensions, 1 do table.insert(dropoutTensorDimensionSizeArray, tensorDimensionSizeArray[i]) end

	local nonDropoutRate = 1 - dropoutRate

	local scalingFactor = 1 / nonDropoutRate

	local resultTensor = AqwamTensorLibrary:copy(pureTensor)

	for i = 1, numberOfData, 1 do

		for j = 1, dimensionSizeAtSecondDimension, 1 do

			if (math.random() > nonDropoutRate) then resultTensor[i][j] = AqwamTensorLibrary:createTensor(dropoutTensorDimensionSizeArray, 0) end

		end

	end

	resultTensor = AqwamTensorLibrary:multiply(resultTensor, scalingFactor)
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local tensor = inputTensorArray[1]

			if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

			tensor:differentiate{firstDerivativeTensor}

		end
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

return DropoutLayers
