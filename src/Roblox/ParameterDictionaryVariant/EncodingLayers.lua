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

local EncodingLayers = {}

local defaultOneHotEncodingBlockMode = "Index"

local defaultNValue = 10000

local function createOneHotTensorFromIndex(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, finalDimensionSize)

	local nextDimension = currentDimension + 1

	local oneHotTensor = {}

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do oneHotTensor[i] = createOneHotTensorFromIndex(tensor[i], dimensionSizeArray, numberOfDimensions, nextDimension, finalDimensionSize) end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			local subTensor = table.create(finalDimensionSize, 0)

			local index = tensor[i]

			if (type(index) ~= "number") then error("The tensor must only have numbers for one hot encoding conversion from indices.") end

			if (subTensor[index]) then subTensor[index] = 1 end

			oneHotTensor[i] = subTensor

		end

	end

	return oneHotTensor

end

local function createOneHotTensorFromKey(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, finalDimensionSize, indexDictionary)

	local nextDimension = currentDimension + 1

	local oneHotTensor = {}

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do oneHotTensor[i] = createOneHotTensorFromKey(tensor[i], dimensionSizeArray, numberOfDimensions, nextDimension, finalDimensionSize, indexDictionary) end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			local subTensor = table.create(finalDimensionSize, 0)

			local key = tensor[i]

			local index = indexDictionary[key]

			if (index) then subTensor[index] = 1 end

			oneHotTensor[i] = subTensor

		end

	end

	return oneHotTensor

end

local function getPositionalEncodingBlockTensor(sequenceLength, k, n)

	local positionalEncodingTensor = {}

	for i = 1, sequenceLength, 2 do

		local exponent = ((2 * i) / sequenceLength)

		local denominator = math.pow(n, exponent)

		positionalEncodingTensor[i] = math.sin(k / denominator)

		positionalEncodingTensor[i + 1] = math.cos(k / denominator)

	end

	return positionalEncodingTensor

end

local function getPositionalEncodingBlockTensorRecursive(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, finalDimensionSize, nValues)

	local nextDimension = currentDimension + 1

	local newTensor = {}

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do newTensor[i] = getPositionalEncodingBlockTensorRecursive(tensor[i], dimensionSizeArray, numberOfDimensions, nextDimension, finalDimensionSize, nValues) end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do newTensor[i] = getPositionalEncodingBlockTensor(finalDimensionSize, i, nValues) end

	end

	return newTensor

end

function EncodingLayers.OneHotEncoding(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local finalDimensionSize = parameterDictionary.finalDimensionSize or parameterDictionary[2]

	local oneHotEncodingMode = parameterDictionary.oneHotEncodingMode or parameterDictionary[3]

	local indexDictionary = parameterDictionary.indexDictionary or parameterDictionary[4]

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	local resultTensor

	if (oneHotEncodingMode == "Index") then

		resultTensor = createOneHotTensorFromIndex(tensor, tensorDimensionSizeArray, numberOfDimensions, 1, finalDimensionSize)

	elseif (oneHotEncodingMode == "Key") then

		if (not indexDictionary) then error("No index dictionary for one hot encoding key mode.") end

		resultTensor = createOneHotTensorFromKey(tensor, tensorDimensionSizeArray, numberOfDimensions, 1, finalDimensionSize, indexDictionary)

	else

		error("Invalid one hot encoding mode.")

	end

	return AutomaticDifferentiationTensor.new({resultTensor, nil, {tensor}})

end

function EncodingLayers.LabelEncoding(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local valueDictionary = parameterDictionary.valueDictionary or parameterDictionary[2]

	local functionToApply = function(x) return (valueDictionary[x] or 0) end

	local resultTensor = AqwamTensorLibrary:applyFunction(functionToApply, tensor)

	return AutomaticDifferentiationTensor.new({resultTensor, nil, {tensor}})

end

function EncodingLayers.PositionEncoding(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local sequenceLength = parameterDictionary.sequenceLength or parameterDictionary[2]

	local nValue = parameterDictionary.nValue or parameterDictionary[3] or defaultNValue

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #dimensionSizeArray

	local resultTensor = getPositionalEncodingBlockTensorRecursive(tensor, dimensionSizeArray, numberOfDimensions, 1, sequenceLength, nValue)

	return AutomaticDifferentiationTensor.new({resultTensor, nil, {tensor}})

end

return EncodingLayers
