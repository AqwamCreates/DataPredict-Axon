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

local DisplayErrorFunctions = require(script.Parent.DisplayErrorFunctions)

local displayFunctionErrorDueToNonObjectCondition = DisplayErrorFunctions.displayFunctionErrorDueToNonObjectCondition

local WeightContainer = {}

WeightContainer.__index = WeightContainer

local defaultLearningRate = 0.01

local defaultSkipMissingGradientTensor = false

local defaultUpdateWeightTensorInPlace = true

local function getValueOrDefaultValue(value, defaultValue)

	if (type(value) == "nil") then return defaultValue end

	return value

end

local function performInPlaceSubtraction(tensorToUpdate, tensorToUseForUpdate, dimensionSizeArray, numberOfDimensions, currentDimension) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local nextDimension = currentDimension + 1

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do performInPlaceSubtraction(tensorToUpdate[i], tensorToUseForUpdate[i], dimensionSizeArray, numberOfDimensions, nextDimension) end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do tensorToUpdate[i] = (tensorToUpdate[i] - tensorToUseForUpdate[i]) end

	end

end

local function performInPlaceAddition(tensorToUpdate, tensorToUseForUpdate, dimensionSizeArray, numberOfDimensions, currentDimension) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local nextDimension = currentDimension + 1

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do performInPlaceAddition(tensorToUpdate[i], tensorToUseForUpdate[i], dimensionSizeArray, numberOfDimensions, nextDimension) end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do tensorToUpdate[i] = (tensorToUpdate[i] + tensorToUseForUpdate[i]) end

	end

end

local function performInPlaceUpdate(inPlaceUpdateFunction, weightTensor, weightLossTensor)
	
	if (type(weightLossTensor) == "number") then error("The weight loss tensor must not be a number in order to use in-place weight update operations.") end

	if (type(weightTensor) == "number") then error("The weight tensor must not be a number in order to use in-place weight update operations.") end

	local weightLossTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightLossTensor)

	local weightTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightTensor)

	local weightLossTensorNumberOfDimensions = #weightLossTensorDimensionSizeArray

	local weightTensorNumberOfDimensions = #weightTensorDimensionSizeArray

	if (weightTensorNumberOfDimensions ~= weightLossTensorNumberOfDimensions) then error("The weight tensor and the weight loss tensor does not have equal number of dimensions. The weight tensor has dimension of " .. weightTensorNumberOfDimensions ..", but weight loss tensor has the dimension of " .. weightLossTensorNumberOfDimensions .. ".") end

	for i, weightTensorDimensionSize in ipairs(weightTensorDimensionSizeArray) do

		local weightLossTensorDimensionSize = weightLossTensorDimensionSizeArray[i]

		if (weightTensorDimensionSize ~= weightLossTensorDimensionSize) then error("The weight tensor and the weight loss tensor does not have equal size at dimension " .. i .. ". The weight tensor has the size of " .. weightTensorDimensionSize ..", but weight loss tensor has the size of " .. weightLossTensorDimensionSize .. ".") end

	end

	inPlaceUpdateFunction(weightTensor, weightLossTensor, weightTensorDimensionSizeArray, weightTensorNumberOfDimensions, 1)
	
end

function WeightContainer.new(parameterDictionary)

	local NewWeightContainer = {}

	setmetatable(NewWeightContainer, WeightContainer)
	
	NewWeightContainer.skipMissingGradientTensor = getValueOrDefaultValue(parameterDictionary.skipMissingGradientTensor or parameterDictionary[1], defaultSkipMissingGradientTensor)

	NewWeightContainer.updateWeightTensorInPlace = getValueOrDefaultValue(parameterDictionary.updateWeightTensorInPlace or parameterDictionary[2], defaultUpdateWeightTensorInPlace)
	
	NewWeightContainer.WeightTensorDataArray = {}
	
	NewWeightContainer.isAnObject = true

	return NewWeightContainer

end

function WeightContainer:setWeightTensorDataArray(parameterDictionary)
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	self.WeightTensorDataArray = parameterDictionary
	
end

function WeightContainer:gradientDescent()
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	local skipMissingGradientTensor = self.skipMissingGradientTensor
	
	local updateWeightTensorInPlace = self.updateWeightTensorInPlace
	
	local WeightTensorDataArray = self.WeightTensorDataArray

	local numberOfElements = #WeightTensorDataArray

	for i = numberOfElements, 1, -1 do

		local WeightTensorDataArray = WeightTensorDataArray[i]

		local automaticDifferentiationTensor = WeightTensorDataArray.automaticDifferentiationTensor or WeightTensorDataArray[1]

		local learningRate =  WeightTensorDataArray.learningRate or WeightTensorDataArray[2] or defaultLearningRate

		local Optimizer = WeightTensorDataArray.Optimizer or WeightTensorDataArray[3]
		
		local Regularizer = WeightTensorDataArray.Regularizer or WeightTensorDataArray[4]

		local firstDerivativeTensor = automaticDifferentiationTensor:getTotalFirstDerivativeTensor{true}
		
		if (firstDerivativeTensor) then
			
			local tensor = automaticDifferentiationTensor:getTensor{true}
			
			if (Regularizer) then

				local regularizationTensor = Regularizer:calculate{tensor}

				firstDerivativeTensor = AqwamTensorLibrary:add(firstDerivativeTensor, regularizationTensor)

			end

			if (Optimizer) then

				firstDerivativeTensor = Optimizer:calculate{learningRate, firstDerivativeTensor, tensor}

			else

				firstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, firstDerivativeTensor)

			end

			if (updateWeightTensorInPlace) then

				performInPlaceUpdate(performInPlaceSubtraction, tensor, firstDerivativeTensor)

			else

				tensor = AqwamTensorLibrary:subtract(tensor, firstDerivativeTensor)

				automaticDifferentiationTensor:setTensor{tensor, true}

			end

			automaticDifferentiationTensor:setTotalFirstDerivativeTensor{nil, true}
			
		else
			
			if (not skipMissingGradientTensor) then error("Unable to find first derivative tensor for ADTensor " .. i .. ".") end
			
		end

	end

end

function WeightContainer:gradientAscent()
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	local skipMissingGradientTensor = self.skipMissingGradientTensor

	local updateWeightTensorInPlace = self.updateWeightTensorInPlace
	
	local WeightTensorDataArray = self.WeightTensorDataArray

	local numberOfElements = #WeightTensorDataArray

	for i = numberOfElements, 1, -1 do

		local WeightTensorDataArray = WeightTensorDataArray[i]

		local automaticDifferentiationTensor = WeightTensorDataArray.automaticDifferentiationTensor or WeightTensorDataArray[1]

		local learningRate = WeightTensorDataArray.learningRate or WeightTensorDataArray[2] or defaultLearningRate

		local Optimizer = WeightTensorDataArray.Optimizer or WeightTensorDataArray[3]
		
		local Regularizer = WeightTensorDataArray.Regularizer or WeightTensorDataArray[4]

		local firstDerivativeTensor = automaticDifferentiationTensor:getTotalFirstDerivativeTensor{true}
		
		if (firstDerivativeTensor) then

			local tensor = automaticDifferentiationTensor:getTensor{true}
			
			if (Regularizer) then

				local regularizationTensor = Regularizer:calculate{tensor}

				firstDerivativeTensor = AqwamTensorLibrary:add(firstDerivativeTensor, regularizationTensor)

			end

			if (Optimizer) then

				firstDerivativeTensor = Optimizer:calculate{learningRate, firstDerivativeTensor, tensor}

			else

				firstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, firstDerivativeTensor)

			end

			if (updateWeightTensorInPlace) then

				performInPlaceUpdate(performInPlaceAddition, tensor, firstDerivativeTensor)

			else

				tensor = AqwamTensorLibrary:add(tensor, firstDerivativeTensor)

				automaticDifferentiationTensor:setTensor{tensor, true}

			end

			automaticDifferentiationTensor:setTotalFirstDerivativeTensor{nil, true}

		else

			if (not skipMissingGradientTensor) then  error("Unable to find first derivative tensor for ADTensor " .. i .. ".") end

		end

	end

end

function WeightContainer:getWeightTensorArray(parameterDictionary)
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	parameterDictionary = parameterDictionary or {}

	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[1]

	local weightTensorArray = {}

	for i, WeightTensorData in ipairs(self.WeightTensorDataArray) do

		local automaticDifferentiationTensor = WeightTensorData[1]

		weightTensorArray[i] = automaticDifferentiationTensor:getTensor{doNotDeepCopy}

	end

	return weightTensorArray

end

function WeightContainer:setWeightTensorArray(parameterDictionary)
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	parameterDictionary = parameterDictionary or {}

	local weightTensorArray = parameterDictionary.weightTensorArray or parameterDictionary[1]

	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[2]

	for i, WeightTensorData in ipairs(self.WeightTensorDataArray) do

		local automaticDifferentiationTensor = WeightTensorData[1]

		automaticDifferentiationTensor:getTensor{weightTensorArray[i], doNotDeepCopy}

	end

end

function WeightContainer:setLearningRate(parameterDictionary)
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local learningRate = parameterDictionary.learningRate or parameterDictionary[1]

	local index = parameterDictionary.index or parameterDictionary[2]

	for i, WeightTensorData in ipairs(self.WeightTensorDataArray) do
		
		if (i == index) or (not index) then
			
			WeightTensorData[i][2] = learningRate
			
		end

	end

end

return WeightContainer
