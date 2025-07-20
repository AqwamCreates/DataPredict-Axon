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

local PaddingLayers = {}

local defaultHeadPaddingDimensionSizeArray = {1, 1}

local defaultTailPaddingDimensionSizeArray = {1, 1}

local defaultValue = 0

local function padArraysToEqualLengths(numberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	local headPaddingNumberOfDimensionsOffset = numberOfDimensions - #headPaddingDimensionSizeArray

	local tailPaddingNumberOfDimensionsOffset = numberOfDimensions - #tailPaddingDimensionSizeArray 

	if (headPaddingNumberOfDimensionsOffset ~= 0) then for i = 1, headPaddingNumberOfDimensionsOffset, 1 do table.insert(headPaddingDimensionSizeArray, 1, 0) end end

	if (tailPaddingNumberOfDimensionsOffset ~= 0) then for i = 1, tailPaddingNumberOfDimensionsOffset, 1 do table.insert(tailPaddingDimensionSizeArray, 1, 0) end end

	return headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray

end

local function incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	for i = #dimensionIndexArray, 1, -1 do

		dimensionIndexArray[i] = dimensionIndexArray[i] + 1

		if (dimensionIndexArray[i] <= dimensionSizeArray[i]) then break end

		dimensionIndexArray[i] = 1

	end

	return dimensionIndexArray

end

local function checkIfDimensionIndexArraysAreEqual(dimensionIndexArray1, dimensionIndexArray2)

	if (#dimensionIndexArray1 ~= #dimensionIndexArray2) then return false end

	for i, index in ipairs(dimensionIndexArray1) do

		if (index ~= dimensionIndexArray2[i]) then return false end

	end

	return true

end

local function getTotalDimensionSize(dimensionSizeArray)

	local totalDimensionSize = 1

	for _, size in ipairs(dimensionSizeArray) do totalDimensionSize = totalDimensionSize * size end

	return totalDimensionSize

end

function PaddingLayers.FastZeroPadding(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or parameterDictionary[2] or defaultHeadPaddingDimensionSizeArray

	local tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or parameterDictionary[3] or defaultTailPaddingDimensionSizeArray
	
	local inputTensorArray = {tensor}
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue(tensor)

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureTensor)

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(tensorNumberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	if (#headPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end

	if (#tailPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end

	local resultTensor = pureTensor

	for dimension = tensorNumberOfDimensions, 1, -1 do

		local resultTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(resultTensor)

		local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

		local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

		if (headPaddingDimensionSize >= 1) then

			local tensorHeadPaddingDimensionSizeArray = table.clone(resultTensorDimensionSizeArray)

			tensorHeadPaddingDimensionSizeArray[dimension] = headPaddingDimensionSize

			local headPaddingTensor = AqwamTensorLibrary:createTensor(tensorHeadPaddingDimensionSizeArray)

			resultTensor = AqwamTensorLibrary:concatenate(headPaddingTensor, resultTensor, dimension)

		end

		if (tailPaddingDimensionSize >= 1) then

			local tensorTailPaddingDimensionSizeArray = table.clone(resultTensorDimensionSizeArray)

			tensorTailPaddingDimensionSizeArray[dimension] = tailPaddingDimensionSize

			local tailPaddingTensor = AqwamTensorLibrary:createTensor(tensorTailPaddingDimensionSizeArray)

			resultTensor = AqwamTensorLibrary:concatenate(resultTensor, tailPaddingTensor, dimension)

		end

	end

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)
		
		local tensor = inputTensorArray[1]

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local originDimensionIndexArray = {}

		local targetDimensionIndexArray = table.clone(tensorDimensionSizeArray)

		for dimension = 1, tensorNumberOfDimensions, 1 do

			originDimensionIndexArray[dimension] = (headPaddingDimensionSizeArray[dimension] or 0) + 1

			targetDimensionIndexArray[dimension] = targetDimensionIndexArray[dimension] + (headPaddingDimensionSizeArray[dimension] or 0)

		end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:extract(firstDerivativeTensor, originDimensionIndexArray, targetDimensionIndexArray)

		tensor:differentiate{chainRuleFirstDerivativeTensor}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function PaddingLayers.FastConstantPadding(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or parameterDictionary[2] or defaultHeadPaddingDimensionSizeArray

	local tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or parameterDictionary[3] or defaultTailPaddingDimensionSizeArray

	local value = parameterDictionary.value or parameterDictionary[4] or defaultValue
	
	local inputTensorArray = {tensor}
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue(tensor)

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureTensor)

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(tensorNumberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	if (#headPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end

	if (#tailPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end

	local chainRuleFirstDerivativeMultiplierValue = 0

	local resultTensor = pureTensor

	for dimension = tensorNumberOfDimensions, 1, -1 do

		local resultTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(resultTensor)

		local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

		local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

		if (headPaddingDimensionSize >= 1) then

			local tensorHeadPaddingDimensionSizeArray = table.clone(resultTensorDimensionSizeArray)

			tensorHeadPaddingDimensionSizeArray[dimension] = headPaddingDimensionSize

			local headPaddingTensor = AqwamTensorLibrary:createTensor(tensorHeadPaddingDimensionSizeArray, value)

			resultTensor = AqwamTensorLibrary:concatenate(headPaddingTensor, resultTensor, dimension)

			chainRuleFirstDerivativeMultiplierValue = chainRuleFirstDerivativeMultiplierValue + getTotalDimensionSize(tensorHeadPaddingDimensionSizeArray)

		end

		if (tailPaddingDimensionSize >= 1) then

			local tensorTailPaddingDimensionSizeArray = table.clone(resultTensorDimensionSizeArray)

			tensorTailPaddingDimensionSizeArray[dimension] = tailPaddingDimensionSize

			local tailPaddingTensor = AqwamTensorLibrary:createTensor(tensorTailPaddingDimensionSizeArray, value)

			resultTensor = AqwamTensorLibrary:concatenate(resultTensor, tailPaddingTensor, dimension)

			chainRuleFirstDerivativeMultiplierValue = chainRuleFirstDerivativeMultiplierValue + getTotalDimensionSize(tensorTailPaddingDimensionSizeArray)

		end

	end

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)
		
		local tensor = inputTensorArray[1]

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local chainRuleFirstDerivativeMultiplierFunction = function(chainRuleFirstDerivativeValue, inputValue, value, chainRuleFirstDerivativeMultiplierValue)

			if (inputValue == value) then chainRuleFirstDerivativeValue = chainRuleFirstDerivativeValue * chainRuleFirstDerivativeMultiplierValue end

			return chainRuleFirstDerivativeValue

		end

		local originDimensionIndexArray = {}

		local targetDimensionIndexArray = table.clone(tensorDimensionSizeArray)

		for dimension = 1, tensorNumberOfDimensions, 1 do

			originDimensionIndexArray[dimension] = (headPaddingDimensionSizeArray[dimension] or 0) + 1

			targetDimensionIndexArray[dimension] = targetDimensionIndexArray[dimension] + (headPaddingDimensionSizeArray[dimension] or 0)

		end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:extract(firstDerivativeTensor, originDimensionIndexArray, targetDimensionIndexArray)

		if (value ~= 0) then

			chainRuleFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(chainRuleFirstDerivativeMultiplierFunction, chainRuleFirstDerivativeTensor, tensor, value, chainRuleFirstDerivativeMultiplierValue)

		end

		tensor:differentiate{chainRuleFirstDerivativeTensor}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function PaddingLayers.FastCircularPadding(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or parameterDictionary[2] or defaultHeadPaddingDimensionSizeArray

	local tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or parameterDictionary[3] or defaultTailPaddingDimensionSizeArray
	
	local inputTensorArray = {tensor}
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue(tensor)

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureTensor)

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(tensorNumberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	if (#headPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end

	if (#tailPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end

	local resultTensor = pureTensor

	for dimension = tensorNumberOfDimensions, 1, -1 do

		local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

		local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

		if (headPaddingDimensionSize >= 1) then

			local resultTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(resultTensor)

			local resultTensorDimensionSize = resultTensorDimensionSizeArray[dimension]

			local resultTensorStartDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

			local resultTensorEndDimensionIndexArray = table.clone(resultTensorDimensionSizeArray)

			resultTensorStartDimensionIndexArray[dimension] = resultTensorDimensionSize

			resultTensorEndDimensionIndexArray[dimension] = resultTensorDimensionSize

			for i = 1, headPaddingDimensionSize, 1 do

				local extractedInputTensor = AqwamTensorLibrary:extract(resultTensor, resultTensorStartDimensionIndexArray, resultTensorEndDimensionIndexArray)

				resultTensor = AqwamTensorLibrary:concatenate(extractedInputTensor, resultTensor, dimension)

			end

		end

		if (tailPaddingDimensionSize >= 1) then

			local resultTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(resultTensor)

			local resultTensorDimensionSize = resultTensorDimensionSizeArray[dimension]

			local resultTensorStartDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

			local resultTensorEndDimensionIndexArray = table.clone(resultTensorDimensionSizeArray)

			local currentIndex = headPaddingDimensionSize + 1

			for i = 1, tailPaddingDimensionSize, 1 do

				resultTensorStartDimensionIndexArray[dimension] = currentIndex

				resultTensorEndDimensionIndexArray[dimension] = currentIndex

				currentIndex = currentIndex + 1

				local extractedInputTensor = AqwamTensorLibrary:extract(resultTensor, resultTensorStartDimensionIndexArray, resultTensorEndDimensionIndexArray)

				resultTensor = AqwamTensorLibrary:concatenate(resultTensor, extractedInputTensor, dimension)

			end

		end

	end

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)
		
		local tensor = inputTensorArray[1]

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local firstDerivativeTensorDimensionSizeArray  = AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor) 

		local firstDerivativeTensorNumberOfDimensions = #firstDerivativeTensorDimensionSizeArray

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray, 0)

		local currentDerivativeTensorDimensionIndexArray = table.create(firstDerivativeTensorNumberOfDimensions, 1)

		currentDerivativeTensorDimensionIndexArray[firstDerivativeTensorNumberOfDimensions] = 0

		local currentInputTensorDimensionIndexArray = {}

		for dimension, inputTensorDimensionSize in ipairs(tensorDimensionSizeArray) do

			local currentTensorDimensionIndex = headPaddingDimensionSizeArray[dimension] % inputTensorDimensionSize

			if (currentTensorDimensionIndex == 0) then currentTensorDimensionIndex = inputTensorDimensionSize end

			currentInputTensorDimensionIndexArray[dimension] = currentTensorDimensionIndex

		end

		currentInputTensorDimensionIndexArray[tensorNumberOfDimensions] = currentInputTensorDimensionIndexArray[tensorNumberOfDimensions] - 1

		local currentChainRuleFirstDerivativeValue

		local initialPartialFirstDerivativeValue

		local newChainRuleFirstDerivativeValue

		repeat

			currentInputTensorDimensionIndexArray = incrementDimensionIndexArray(currentInputTensorDimensionIndexArray, tensorDimensionSizeArray)

			currentDerivativeTensorDimensionIndexArray = incrementDimensionIndexArray(currentDerivativeTensorDimensionIndexArray, firstDerivativeTensorDimensionSizeArray)

			currentChainRuleFirstDerivativeValue = AqwamTensorLibrary:getValue(chainRuleFirstDerivativeTensor, currentInputTensorDimensionIndexArray)

			initialPartialFirstDerivativeValue = AqwamTensorLibrary:getValue(firstDerivativeTensor, currentDerivativeTensorDimensionIndexArray)  

			newChainRuleFirstDerivativeValue = currentChainRuleFirstDerivativeValue + initialPartialFirstDerivativeValue

			AqwamTensorLibrary:setValue(chainRuleFirstDerivativeTensor, newChainRuleFirstDerivativeValue, currentInputTensorDimensionIndexArray)

		until checkIfDimensionIndexArraysAreEqual(currentDerivativeTensorDimensionIndexArray, firstDerivativeTensorDimensionSizeArray)

		tensor:differentiate{chainRuleFirstDerivativeTensor}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function PaddingLayers.FastReplicationPaddingBlock(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or parameterDictionary[2] or defaultHeadPaddingDimensionSizeArray

	local tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or parameterDictionary[3] or defaultTailPaddingDimensionSizeArray
	
	local inputTensorArray = {tensor}
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue(tensor)

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureTensor)

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(tensorNumberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	if (#headPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end

	if (#tailPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end

	local resultTensor = pureTensor

	for dimension = tensorNumberOfDimensions, 1, -1 do

		local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

		local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

		if (headPaddingDimensionSize >= 1) then

			local resultTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(resultTensor)

			local resultTensorDimensionSize = resultTensorDimensionSizeArray[dimension]

			local resultTensorStartDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

			local resultTensorEndDimensionIndexArray = table.clone(resultTensorDimensionSizeArray)

			resultTensorStartDimensionIndexArray[dimension] = 1

			resultTensorEndDimensionIndexArray[dimension] = 1

			local extractedInputTensor = AqwamTensorLibrary:extract(resultTensor, resultTensorStartDimensionIndexArray, resultTensorEndDimensionIndexArray)

			for i = 1, headPaddingDimensionSize, 1 do resultTensor = AqwamTensorLibrary:concatenate(extractedInputTensor, resultTensor, dimension) end

		end

		if (tailPaddingDimensionSize >= 1) then

			local resultTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(resultTensor)

			local resultTensorDimensionSize = resultTensorDimensionSizeArray[dimension]

			local resultTensorStartDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

			local resultTensorEndDimensionIndexArray = table.clone(resultTensorDimensionSizeArray)

			resultTensorStartDimensionIndexArray[dimension] = resultTensorDimensionSize

			resultTensorEndDimensionIndexArray[dimension] = resultTensorDimensionSize

			local extractedInputTensor = AqwamTensorLibrary:extract(resultTensor, resultTensorStartDimensionIndexArray, resultTensorEndDimensionIndexArray)

			for i = 1, tailPaddingDimensionSize, 1 do resultTensor = AqwamTensorLibrary:concatenate(resultTensor, extractedInputTensor, dimension) end

		end

	end

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)
		
		local tensor = inputTensorArray[1]
		
		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local tensorNumberOfDimensions = #tensorDimensionSizeArray

		local originDimensionIndexArray = {}

		local targetDimensionIndexArray = table.clone(tensorDimensionSizeArray)

		for dimension = 1, tensorNumberOfDimensions, 1 do

			originDimensionIndexArray[dimension] = (headPaddingDimensionSizeArray[dimension] or 0) + 1

			targetDimensionIndexArray[dimension] = targetDimensionIndexArray[dimension] + (headPaddingDimensionSizeArray[dimension] or 0)

		end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:extract(firstDerivativeTensor, originDimensionIndexArray, targetDimensionIndexArray)

		local originExtractionDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

		for dimension = 1, tensorNumberOfDimensions, 1 do

			local tensorDimensionSize = tensorDimensionSizeArray[dimension]

			local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

			local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

			if (headPaddingDimensionSize >= 1) then -- Head gradient edge cases.

				if (tensorDimensionSize > 1) then

					local targetExtractionDimensionIndexArray = table.clone(tensorDimensionSizeArray)

					targetExtractionDimensionIndexArray[dimension] = 1

					local extractedChainRuleFirstDerivativeHeadTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeTensor, originExtractionDimensionIndexArray, targetExtractionDimensionIndexArray)

					extractedChainRuleFirstDerivativeHeadTensor = AqwamTensorLibrary:multiply(extractedChainRuleFirstDerivativeHeadTensor, headPaddingDimensionSize)

					local remainingChainRuleFirstDerivativeTensorHeadDimensionIndexArray = table.clone(originExtractionDimensionIndexArray)

					remainingChainRuleFirstDerivativeTensorHeadDimensionIndexArray[dimension] = remainingChainRuleFirstDerivativeTensorHeadDimensionIndexArray[dimension] + 1

					local remainingChainRuleFirstDerivativeHeadTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeTensor, remainingChainRuleFirstDerivativeTensorHeadDimensionIndexArray, tensorDimensionSizeArray)

					chainRuleFirstDerivativeTensor = AqwamTensorLibrary:concatenate(extractedChainRuleFirstDerivativeHeadTensor, remainingChainRuleFirstDerivativeHeadTensor, dimension)		

				else

					chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(chainRuleFirstDerivativeTensor, headPaddingDimensionSize)

				end

			end

			if (tailPaddingDimensionSize >= 1) then -- Tail gradient edge cases.

				if (tensorDimensionSize > 1) then

					local remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray = table.clone(tensorDimensionSizeArray)

					remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray[dimension] = remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray[dimension] - 1

					local extractedChainRuleFirstDerivativeTensorTailDimensionIndexArray = table.clone(originExtractionDimensionIndexArray)

					extractedChainRuleFirstDerivativeTensorTailDimensionIndexArray[dimension] = tensorDimensionSizeArray[dimension]

					local remainingChainRuleFirstDerivativeTailTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeTensor, originExtractionDimensionIndexArray, remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray)

					local targetChainRuleFirstDerivativeTailTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeTensor, extractedChainRuleFirstDerivativeTensorTailDimensionIndexArray, tensorDimensionSizeArray)

					targetChainRuleFirstDerivativeTailTensor = AqwamTensorLibrary:multiply(targetChainRuleFirstDerivativeTailTensor, tailPaddingDimensionSize)

					chainRuleFirstDerivativeTensor = AqwamTensorLibrary:concatenate(remainingChainRuleFirstDerivativeTailTensor, targetChainRuleFirstDerivativeTailTensor, dimension)

				else

					chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(chainRuleFirstDerivativeTensor, tailPaddingDimensionSize)

				end

			end

		end

		tensor:differentiate{chainRuleFirstDerivativeTensor}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function PaddingLayers.FastReflectionPaddingBlock(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or parameterDictionary[2] or defaultHeadPaddingDimensionSizeArray

	local tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or parameterDictionary[3] or defaultTailPaddingDimensionSizeArray
	
	local inputTensorArray = {tensor}
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue(tensor)

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureTensor)

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(tensorNumberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	if (#headPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end

	if (#tailPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end

	for dimension = 1, tensorNumberOfDimensions, 1 do

		local tensorDimensionSize = tensorDimensionSizeArray[dimension]

		local headDimensionSize = headPaddingDimensionSizeArray[dimension]

		local tailDimensionSize = tailPaddingDimensionSizeArray[dimension]

		local errorStringEnding = " must not be greater or equal to the dimension size of " .. tensorDimensionSize .. " from the input tensor."

		if (headDimensionSize >= tensorDimensionSize) then error("The head padding dimension size of " .. headDimensionSize .. " at dimension " .. dimension .. errorStringEnding) end

		if (tailDimensionSize >= tensorDimensionSize) then error("The tail padding dimension size of " .. tailDimensionSize .. " at dimension " .. dimension .. errorStringEnding) end

	end

	local resultTensor = pureTensor

	for dimension = tensorNumberOfDimensions, 1, -1 do

		local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

		local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

		if (headPaddingDimensionSize >= 1) then

			local resultTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(resultTensor)

			local resultTensorDimensionSize = resultTensorDimensionSizeArray[dimension]

			local resultTensorStartDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

			local resultTensorEndDimensionIndexArray = table.clone(resultTensorDimensionSizeArray)

			local startingIndex = 1

			for i = 1, headPaddingDimensionSize, 1 do

				local currentIndex = startingIndex + i 

				resultTensorStartDimensionIndexArray[dimension] = currentIndex

				resultTensorEndDimensionIndexArray[dimension] = currentIndex

				startingIndex = startingIndex + 1

				local extractedInputTensor = AqwamTensorLibrary:extract(resultTensor, resultTensorStartDimensionIndexArray, resultTensorEndDimensionIndexArray)

				resultTensor = AqwamTensorLibrary:concatenate(extractedInputTensor, resultTensor, dimension) 

			end

		end

		if (tailPaddingDimensionSize >= 1) then

			local resultTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(resultTensor)

			local resultTensorDimensionSize = resultTensorDimensionSizeArray[dimension]

			local resultTensorStartDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

			local resultTensorEndDimensionIndexArray = table.clone(resultTensorDimensionSizeArray)

			local startingIndex = resultTensorDimensionSize

			for i = 1, tailPaddingDimensionSize, 1 do

				local currentIndex = startingIndex - i

				resultTensorStartDimensionIndexArray[dimension] = currentIndex

				resultTensorEndDimensionIndexArray[dimension] = currentIndex

				local extractedInputTensor = AqwamTensorLibrary:extract(resultTensor, resultTensorStartDimensionIndexArray, resultTensorEndDimensionIndexArray)

				resultTensor = AqwamTensorLibrary:concatenate(resultTensor, extractedInputTensor, dimension) 

			end

		end

	end

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)
		
		local tensor = inputTensorArray[1]
		
		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local originDimensionIndexArray = {}

		local targetDimensionIndexArray = table.clone(tensorDimensionSizeArray)

		for dimension = 1, tensorNumberOfDimensions, 1 do

			originDimensionIndexArray[dimension] = (headPaddingDimensionSizeArray[dimension] or 0) + 1

			targetDimensionIndexArray[dimension] = targetDimensionIndexArray[dimension] + (headPaddingDimensionSizeArray[dimension] or 0)

		end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:extract(firstDerivativeTensor, originDimensionIndexArray, targetDimensionIndexArray)

		local chainRuleFirstDerivativeMultiplierTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		local originExtractionDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

		for dimension = 1, tensorNumberOfDimensions, 1 do -- Gradient edge cases.

			local inputTensorDimensionSize = tensorDimensionSizeArray[dimension]

			local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

			local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

			if (headPaddingDimensionSize >= 1) then

				if (inputTensorDimensionSize > 1) then

					local remainingExtractionDimensionIndexArray = table.clone(tensorDimensionSizeArray)

					remainingExtractionDimensionIndexArray[dimension] = 1

					local extractedChainRuleFirstDerivativeHeadTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeMultiplierTensor, originExtractionDimensionIndexArray, remainingExtractionDimensionIndexArray)

					local targetChainRuleFirstDerivativeTensorHeadDimensionIndexArray = table.clone(originExtractionDimensionIndexArray)

					targetChainRuleFirstDerivativeTensorHeadDimensionIndexArray[dimension] = targetChainRuleFirstDerivativeTensorHeadDimensionIndexArray[dimension] + 1

					local targetChainRuleFirstDerivativeHeadTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeMultiplierTensor, targetChainRuleFirstDerivativeTensorHeadDimensionIndexArray, tensorDimensionSizeArray)

					targetChainRuleFirstDerivativeHeadTensor = AqwamTensorLibrary:add(targetChainRuleFirstDerivativeHeadTensor, 1)

					chainRuleFirstDerivativeMultiplierTensor = AqwamTensorLibrary:concatenate(extractedChainRuleFirstDerivativeHeadTensor, targetChainRuleFirstDerivativeHeadTensor, dimension)	

				else

					chainRuleFirstDerivativeMultiplierTensor = AqwamTensorLibrary:add(chainRuleFirstDerivativeMultiplierTensor, 1)

				end

			end

			if (tailPaddingDimensionSize >= 1) then -- Tail gradient edge cases.

				if (inputTensorDimensionSize > 1) then

					local remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray = table.clone(tensorDimensionSizeArray)

					remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray[dimension] = remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray[dimension] - 1

					local extractedChainRuleFirstDerivativeTensorTailDimensionIndexArray = table.clone(originExtractionDimensionIndexArray)

					extractedChainRuleFirstDerivativeTensorTailDimensionIndexArray[dimension] = tensorDimensionSizeArray[dimension]

					local targetChainRuleFirstDerivativeTailTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeMultiplierTensor, originExtractionDimensionIndexArray, remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray)

					local remainingChainRuleFirstDerivativeTailTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeMultiplierTensor, extractedChainRuleFirstDerivativeTensorTailDimensionIndexArray, tensorDimensionSizeArray)

					targetChainRuleFirstDerivativeTailTensor = AqwamTensorLibrary:add(targetChainRuleFirstDerivativeTailTensor, 1)

					chainRuleFirstDerivativeMultiplierTensor = AqwamTensorLibrary:concatenate(targetChainRuleFirstDerivativeTailTensor, remainingChainRuleFirstDerivativeTailTensor, dimension)

				else

					chainRuleFirstDerivativeMultiplierTensor = AqwamTensorLibrary:add(chainRuleFirstDerivativeMultiplierTensor, tailPaddingDimensionSize)

				end

			end

		end

		chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(chainRuleFirstDerivativeTensor, chainRuleFirstDerivativeMultiplierTensor)

		tensor:differentiate{chainRuleFirstDerivativeTensor}

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

---------------------------------------------------------------------------

function PaddingLayers.ZeroPadding(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or parameterDictionary[2] or defaultHeadPaddingDimensionSizeArray

	local tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or parameterDictionary[3] or defaultTailPaddingDimensionSizeArray
	
	tensor = AutomaticDifferentiationTensor.coerce{tensor}
	
	local inputTensorArray = {tensor}

	local tensorDimensionSizeArray = tensor:getDimensionSizeArray()

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(tensorNumberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	if (#headPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end

	if (#tailPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end

	local resultTensor = tensor

	for dimension = tensorNumberOfDimensions, 1, -1 do

		local resultTensorDimensionSizeArray = resultTensor:getDimensionSizeArray()

		local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

		local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

		if (headPaddingDimensionSize >= 1) then

			local tensorHeadPaddingDimensionSizeArray = table.clone(resultTensorDimensionSizeArray)

			tensorHeadPaddingDimensionSizeArray[dimension] = headPaddingDimensionSize

			local headPaddingTensor = AqwamTensorLibrary:createTensor(tensorHeadPaddingDimensionSizeArray)

			resultTensor = AutomaticDifferentiationTensor.concatenate{headPaddingTensor, resultTensor, dimension}

		end

		if (tailPaddingDimensionSize >= 1) then

			local tensorTailPaddingDimensionSizeArray = table.clone(resultTensorDimensionSizeArray)

			tensorTailPaddingDimensionSizeArray[dimension] = tailPaddingDimensionSize

			local tailPaddingTensor = AqwamTensorLibrary:createTensor(tensorTailPaddingDimensionSizeArray)

			resultTensor = AutomaticDifferentiationTensor.concatenate{resultTensor, tailPaddingTensor, dimension}

		end

	end

	return resultTensor

end

function PaddingLayers.ConstantPadding(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or parameterDictionary[2] or defaultHeadPaddingDimensionSizeArray

	local tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or parameterDictionary[3] or defaultTailPaddingDimensionSizeArray

	local value = parameterDictionary.value or parameterDictionary[4] or defaultValue
	
	tensor = AutomaticDifferentiationTensor.coerce{tensor}

	local tensorDimensionSizeArray = tensor:getDimensionSizeArray()

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(tensorNumberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	if (#headPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end

	if (#tailPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end

	local chainRuleFirstDerivativeMultiplierValue = 0

	local resultTensor = tensor

	for dimension = tensorNumberOfDimensions, 1, -1 do

		local resultTensorDimensionSizeArray = resultTensor:getDimensionSizeArray()

		local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

		local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

		if (headPaddingDimensionSize >= 1) then

			local tensorHeadPaddingDimensionSizeArray = table.clone(resultTensorDimensionSizeArray)

			tensorHeadPaddingDimensionSizeArray[dimension] = headPaddingDimensionSize

			local headPaddingTensor = AqwamTensorLibrary:createTensor(tensorHeadPaddingDimensionSizeArray, value)

			resultTensor = AutomaticDifferentiationTensor.concatenate{headPaddingTensor, resultTensor, dimension}

			chainRuleFirstDerivativeMultiplierValue = chainRuleFirstDerivativeMultiplierValue + getTotalDimensionSize(tensorHeadPaddingDimensionSizeArray)

		end

		if (tailPaddingDimensionSize >= 1) then

			local tensorTailPaddingDimensionSizeArray = table.clone(resultTensorDimensionSizeArray)

			tensorTailPaddingDimensionSizeArray[dimension] = tailPaddingDimensionSize

			local tailPaddingTensor = AqwamTensorLibrary:createTensor(tensorTailPaddingDimensionSizeArray, value)

			resultTensor = AutomaticDifferentiationTensor.concatenate{resultTensor, tailPaddingTensor, dimension}

			chainRuleFirstDerivativeMultiplierValue = chainRuleFirstDerivativeMultiplierValue + getTotalDimensionSize(tensorTailPaddingDimensionSizeArray)

		end

	end

	return resultTensor

end

function PaddingLayers.CircularPadding(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or parameterDictionary[2] or defaultHeadPaddingDimensionSizeArray

	local tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or parameterDictionary[3] or defaultTailPaddingDimensionSizeArray
	
	tensor = AutomaticDifferentiationTensor.coerce{tensor}

	local tensorDimensionSizeArray = tensor:getDimensionSizeArray()

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(tensorNumberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	if (#headPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end

	if (#tailPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end

	local resultTensor = tensor

	for dimension = tensorNumberOfDimensions, 1, -1 do

		local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

		local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

		if (headPaddingDimensionSize >= 1) then

			local resultTensorDimensionSizeArray = resultTensor:getDimensionSizeArray()

			local resultTensorDimensionSize = resultTensorDimensionSizeArray[dimension]

			local resultTensorStartDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

			local resultTensorEndDimensionIndexArray = table.clone(resultTensorDimensionSizeArray)

			resultTensorStartDimensionIndexArray[dimension] = resultTensorDimensionSize

			resultTensorEndDimensionIndexArray[dimension] = resultTensorDimensionSize

			for i = 1, headPaddingDimensionSize, 1 do

				local extractedInputTensor = resultTensor:extract{resultTensorStartDimensionIndexArray, resultTensorEndDimensionIndexArray}

				resultTensor = AutomaticDifferentiationTensor.concatenate{extractedInputTensor, resultTensor, dimension}

			end

		end

		if (tailPaddingDimensionSize >= 1) then

			local resultTensorDimensionSizeArray = resultTensor:getDimensionSizeArray()

			local resultTensorDimensionSize = resultTensorDimensionSizeArray[dimension]

			local resultTensorStartDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

			local resultTensorEndDimensionIndexArray = table.clone(resultTensorDimensionSizeArray)

			local currentIndex = headPaddingDimensionSize + 1

			for i = 1, tailPaddingDimensionSize, 1 do

				resultTensorStartDimensionIndexArray[dimension] = currentIndex

				resultTensorEndDimensionIndexArray[dimension] = currentIndex

				currentIndex = currentIndex + 1

				local extractedInputTensor = resultTensor:extract{resultTensorStartDimensionIndexArray, resultTensorEndDimensionIndexArray}

				resultTensor = AutomaticDifferentiationTensor.concatenate{resultTensor, extractedInputTensor, dimension}

			end

		end

	end

	return resultTensor

end

function PaddingLayers.ReplicationPadding(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or parameterDictionary[2] or defaultHeadPaddingDimensionSizeArray

	local tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or parameterDictionary[3] or defaultTailPaddingDimensionSizeArray
	
	tensor = AutomaticDifferentiationTensor.coerce{tensor}

	local tensorDimensionSizeArray = tensor:getDimensionSizeArray()

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(tensorNumberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	if (#headPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end

	if (#tailPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end

	local resultTensor = tensor

	for dimension = tensorNumberOfDimensions, 1, -1 do

		local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

		local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

		if (headPaddingDimensionSize >= 1) then

			local resultTensorDimensionSizeArray = resultTensor:getDimensionSizeArray()

			local resultTensorDimensionSize = resultTensorDimensionSizeArray[dimension]

			local resultTensorStartDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

			local resultTensorEndDimensionIndexArray = table.clone(resultTensorDimensionSizeArray)

			resultTensorStartDimensionIndexArray[dimension] = 1

			resultTensorEndDimensionIndexArray[dimension] = 1

			local extractedInputTensor = resultTensor:extract{resultTensorStartDimensionIndexArray, resultTensorEndDimensionIndexArray}

			for i = 1, headPaddingDimensionSize, 1 do resultTensor = AutomaticDifferentiationTensor.concatenate{extractedInputTensor, resultTensor, dimension} end

		end

		if (tailPaddingDimensionSize >= 1) then

			local resultTensorDimensionSizeArray = resultTensor:getDimensionSizeArray()

			local resultTensorDimensionSize = resultTensorDimensionSizeArray[dimension]

			local resultTensorStartDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

			local resultTensorEndDimensionIndexArray = table.clone(resultTensorDimensionSizeArray)

			resultTensorStartDimensionIndexArray[dimension] = resultTensorDimensionSize

			resultTensorEndDimensionIndexArray[dimension] = resultTensorDimensionSize

			local extractedInputTensor = resultTensor:extract{resultTensorStartDimensionIndexArray, resultTensorEndDimensionIndexArray}

			for i = 1, tailPaddingDimensionSize, 1 do resultTensor = AutomaticDifferentiationTensor.concatenate{resultTensor, extractedInputTensor, dimension} end

		end

	end

	return resultTensor

end

function PaddingLayers.ReflectionPadding(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or parameterDictionary[2] or defaultHeadPaddingDimensionSizeArray

	local tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or parameterDictionary[3] or defaultTailPaddingDimensionSizeArray
	
	tensor = AutomaticDifferentiationTensor.coerce{tensor}

	local tensorDimensionSizeArray = tensor:getDimensionSizeArray()

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(tensorNumberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	if (#headPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end

	if (#tailPaddingDimensionSizeArray > tensorNumberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end

	for dimension = 1, tensorNumberOfDimensions, 1 do

		local tensorDimensionSize = tensorDimensionSizeArray[dimension]

		local headDimensionSize = headPaddingDimensionSizeArray[dimension]

		local tailDimensionSize = tailPaddingDimensionSizeArray[dimension]

		local errorStringEnding = " must not be greater or equal to the dimension size of " .. tensorDimensionSize .. " from the input tensor."

		if (headDimensionSize >= tensorDimensionSize) then error("The head padding dimension size of " .. headDimensionSize .. " at dimension " .. dimension .. errorStringEnding) end

		if (tailDimensionSize >= tensorDimensionSize) then error("The tail padding dimension size of " .. tailDimensionSize .. " at dimension " .. dimension .. errorStringEnding) end

	end

	local resultTensor = tensor

	for dimension = tensorNumberOfDimensions, 1, -1 do
		
		print(dimension)

		local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

		local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

		if (headPaddingDimensionSize >= 1) then

			local resultTensorDimensionSizeArray = resultTensor:getDimensionSizeArray()

			local resultTensorDimensionSize = resultTensorDimensionSizeArray[dimension]

			local resultTensorStartDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

			local resultTensorEndDimensionIndexArray = table.clone(resultTensorDimensionSizeArray)

			local startingIndex = 1

			for i = 1, headPaddingDimensionSize, 1 do

				local currentIndex = startingIndex + i 

				resultTensorStartDimensionIndexArray[dimension] = currentIndex

				resultTensorEndDimensionIndexArray[dimension] = currentIndex

				startingIndex = startingIndex + 1

				local extractedInputTensor = resultTensor:extract{resultTensorStartDimensionIndexArray, resultTensorEndDimensionIndexArray}

				resultTensor = AutomaticDifferentiationTensor.concatenate{extractedInputTensor, resultTensor, dimension}

			end

		end

		if (tailPaddingDimensionSize >= 1) then

			local resultTensorDimensionSizeArray = resultTensor:getDimensionSizeArray()

			local resultTensorDimensionSize = resultTensorDimensionSizeArray[dimension]

			local resultTensorStartDimensionIndexArray = table.create(tensorNumberOfDimensions, 1)

			local resultTensorEndDimensionIndexArray = table.clone(resultTensorDimensionSizeArray)

			local startingIndex = resultTensorDimensionSize

			for i = 1, tailPaddingDimensionSize, 1 do

				local currentIndex = startingIndex - i

				resultTensorStartDimensionIndexArray[dimension] = currentIndex

				resultTensorEndDimensionIndexArray[dimension] = currentIndex

				local extractedInputTensor = resultTensor:extract{resultTensorStartDimensionIndexArray, resultTensorEndDimensionIndexArray}

				resultTensor = AutomaticDifferentiationTensor.concatenate{resultTensor, extractedInputTensor, dimension}

			end

		end

	end

	return resultTensor

end

return PaddingLayers
