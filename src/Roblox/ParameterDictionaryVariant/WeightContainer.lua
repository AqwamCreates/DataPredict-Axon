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

local WeightContainer = {}

WeightContainer.__index = WeightContainer

local defaultLearningRate = 0.01

local defaultUpdateWeightTensorInPlace = true

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

	NewWeightContainer.TensorAndOptimizerArrayArray = parameterDictionary

	NewWeightContainer.updateWeightTensorInPlace = parameterDictionary.updateWeightTensorInPlace or defaultUpdateWeightTensorInPlace

	return NewWeightContainer

end

function WeightContainer:gradientDescent()

	local TensorAndOptimizerArrayArray = self.TensorAndOptimizerArrayArray

	local numberOfElements = #TensorAndOptimizerArrayArray
	
	local updateWeightTensorInPlace = self.updateWeightTensorInPlace

	for i = numberOfElements, 1, -1 do

		local TensorAndOptimizerArray = TensorAndOptimizerArrayArray[i]

		local automaticDifferentiationTensor = TensorAndOptimizerArray.automaticDifferentiationTensor or TensorAndOptimizerArray[1]

		local learningRate =  TensorAndOptimizerArray.learningRate or TensorAndOptimizerArray[2] or defaultLearningRate

		local Optimizer = TensorAndOptimizerArray.Optimizer or TensorAndOptimizerArray[3]

		local firstDerivativeTensor = automaticDifferentiationTensor:getTotalFirstDerivativeTensor()

		local tensor = automaticDifferentiationTensor:getTensor({doNotDeepCopy = true})

		local optimizedFirstDerivativeTensor

		if (not firstDerivativeTensor) then error("Unable to find first derivative tensor for ADTensor " .. i .. ".") end

		if (Optimizer) then

			optimizedFirstDerivativeTensor = Optimizer:calculate{learningRate, firstDerivativeTensor}

		else

			optimizedFirstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, firstDerivativeTensor)

		end
		
		if (updateWeightTensorInPlace) then

			performInPlaceUpdate(performInPlaceSubtraction, tensor, optimizedFirstDerivativeTensor)
			
		else

			tensor = AqwamTensorLibrary:subtract(tensor, optimizedFirstDerivativeTensor)

			automaticDifferentiationTensor:setTensor{tensor, true}

		end

		automaticDifferentiationTensor:setTotalFirstDerivativeTensor{nil, true}

	end

end

function WeightContainer:gradientAscent()

	local TensorAndOptimizerArrayArray = self.TensorAndOptimizerArrayArray

	local numberOfElements = #TensorAndOptimizerArrayArray

	local updateWeightTensorInPlace = self.updateWeightTensorInPlace

	for i = numberOfElements, 1, -1 do

		local TensorAndOptimizerArray = TensorAndOptimizerArrayArray[i]

		local automaticDifferentiationTensor = TensorAndOptimizerArray.automaticDifferentiationTensor or TensorAndOptimizerArray[1]

		local learningRate =  TensorAndOptimizerArray.learningRate or TensorAndOptimizerArray[2] or defaultLearningRate

		local Optimizer = TensorAndOptimizerArray.Optimizer or TensorAndOptimizerArray[3]

		local firstDerivativeTensor = automaticDifferentiationTensor:getTotalFirstDerivativeTensor()

		local tensor = automaticDifferentiationTensor:getTensor({doNotDeepCopy = true})

		local optimizedFirstDerivativeTensor

		if (not firstDerivativeTensor) then error("Unable to find first derivative tensor for ADTensor " .. i .. ".") end

		if (Optimizer) then

			optimizedFirstDerivativeTensor = Optimizer:calculate{learningRate, firstDerivativeTensor}

		else

			optimizedFirstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, firstDerivativeTensor)

		end

		if (updateWeightTensorInPlace) then

			performInPlaceUpdate(performInPlaceAddition, tensor, optimizedFirstDerivativeTensor)

		else

			tensor = AqwamTensorLibrary:add(tensor, optimizedFirstDerivativeTensor)

			automaticDifferentiationTensor:setTensor{tensor, true}

		end

		automaticDifferentiationTensor:setTotalFirstDerivativeTensor{nil, true}

	end

end

function WeightContainer:getTensorArray(parameterDictionary)

	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[1]

	local tensorArray = {}

	for i, TensorAndOptimizerArray in ipairs(self.TensorAndOptimizerArrayArray) do

		local automaticDifferentiationTensor = TensorAndOptimizerArray[1]

		tensorArray[i] = automaticDifferentiationTensor:getTensor(doNotDeepCopy)

	end

	return tensorArray

end

function WeightContainer:setTensorArray(parameterDictionary)

	local tensorArray = parameterDictionary.tensorArray or parameterDictionary[1]

	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[2]

	for i, TensorAndOptimizerArray in ipairs(self.TensorAndOptimizerArrayArray) do

		local automaticDifferentiationTensor = TensorAndOptimizerArray[1]

		automaticDifferentiationTensor:getTensor(tensorArray[i], doNotDeepCopy)

	end

end

return WeightContainer