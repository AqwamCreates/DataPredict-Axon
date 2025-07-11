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

local AHAAutomaticDifferentiationTensor = {}

local function showFunctionErrorDueToNonObjectCondition(showError)

	if (showError) then error("This function can only be called if it is an object.") end

end

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else

		copy = original

	end

	return copy

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

local function createOriginalDimensionArray(targetDimensionArray)

	local originalDimensionArray = {}

	local originalDimension = 1

	for i, targetDimension in ipairs(targetDimensionArray) do

		originalDimensionArray[targetDimension] = originalDimension

		originalDimension = originalDimension + 1

	end

	return originalDimensionArray

end

function AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local isAutomaticDifferentiationTensor = pcall(function()

		tensor:isAutomaticDifferentiationTensor()

	end)

	return isAutomaticDifferentiationTensor

end

function AHAAutomaticDifferentiationTensor:fetchValue(automaticDifferentiationTensor) -- DO NOT REMOVE THIS. I REPEAT. DO NOT REMOVE THIS AT ALL COSTS! THIS IS BECAUSE YOUR TENSOR LIBRARY CANNOT HANDLE AUTOMATIC DIFFERENTIATION TENSOR OBJECTS STORING A SCALAR VALUE!

	if (type(automaticDifferentiationTensor) ~= "table") then return automaticDifferentiationTensor end

	return automaticDifferentiationTensor.tensor or automaticDifferentiationTensor

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local self = setmetatable({}, AHAAutomaticDifferentiationTensor)

	self.tensor = parameterDictionary.tensor or parameterDictionary[1]

	self.PartialFirstDerivativeFunction = parameterDictionary.PartialFirstDerivativeFunction or parameterDictionary[2]

	self.inputTensorArray = parameterDictionary.inputTensorArray or parameterDictionary[3]

	self.totalFirstDerivativeTensor = nil

	self.isAnObject = true

	return self

end

function AHAAutomaticDifferentiationTensor.radian(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local inputTensorArray = {tensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(math.rad, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local radiansPerDegree = math.pi / 180

		tensor:differentiate{AqwamTensorLibrary:multiply(radiansPerDegree, firstDerivativeTensor)}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.degree(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local inputTensorArray = {tensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(math.deg, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local degreesPerRadian = 180 / math.pi

		tensor:differentiate{AqwamTensorLibrary:multiply(degreesPerRadian, firstDerivativeTensor)}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.sin(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local inputTensorArray = {tensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(math.sin, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(math.cos, tensor)

		tensor:differentiate{AqwamTensorLibrary:multiply(partialDerivativeTensor, firstDerivativeTensor)}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.cos(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local inputTensorArray = {tensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(math.cos, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local partialDerivativeFunctionToApply = function (radian) return -math.sin(radian) end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate{AqwamTensorLibrary:multiply(partialDerivativeTensor, firstDerivativeTensor)}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.tan(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local inputTensorArray = {tensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(math.tan, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local partialDerivativeFunctionToApply = function (radian) return math.pow((1 / math.cos(radian)), 2) end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate{AqwamTensorLibrary:multiply(partialDerivativeTensor, firstDerivativeTensor)}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.exponent(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local inputTensorArray = {tensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(math.exp, tensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		tensor:differentiate{AqwamTensorLibrary:multiply(resultTensor, firstDerivativeTensor)}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction,inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.logarithm(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local numberTensor = parameterDictionary.numberTensor or parameterDictionary[1]

	local baseTensor = parameterDictionary.baseTensor or parameterDictionary[2]

	local inputTensorArray = {numberTensor, baseTensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(math.log, numberTensor, baseTensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local numberTensor = inputTensorArray[1]

		local baseTensor = inputTensorArray[2]

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{numberTensor} then

			local numberTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(numberTensor)

			local collapsedDerivativeTensor = collapseTensor(firstDerivativeTensor, numberTensorDimensionSizeArray)

			local partialDerivativeTensor

			if (baseTensor) then

				local partialDerivativeFunctionToApply = function (number, base) return (1 / (number * math.log(base))) end

				partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, numberTensor, baseTensor)

			else

				local partialDerivativeFunctionToApply = function (number) return (1 / number) end

				partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, numberTensor)

			end

			numberTensor:differentiate{AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)}

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{baseTensor} then

			local baseTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(baseTensor)

			local collapsedDerivativeTensor = collapseTensor(firstDerivativeTensor, baseTensorDimensionSizeArray)

			local partialDerivativeFunctionToApply = function (number, base) return -(math.log(number) / (base * math.pow(math.log(base), 2))) end

			local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, numberTensor, baseTensorDimensionSizeArray)

			baseTensor:differentiate{AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)}

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.clamp(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local lowerBoundTensor = parameterDictionary.lowerBoundTensor or parameterDictionary[2]

	local upperBoundTensor = parameterDictionary.upperBoundTensor or parameterDictionary[3]

	local inputTensorArray = {tensor, lowerBoundTensor, upperBoundTensor}

	lowerBoundTensor = lowerBoundTensor or -math.huge

	upperBoundTensor = upperBoundTensor or math.huge

	local resultTensor = AqwamTensorLibrary:applyFunction(math.clamp, tensor, lowerBoundTensor, upperBoundTensor)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

		local lowerBoundTensor = inputTensorArray[2]

		local upperBoundTensor = inputTensorArray[3]

		local functionToApply = function(value, derivative, lowerBoundValue, upperBoundValue) if ((value >= lowerBoundValue) and (value <= upperBoundValue)) then return value else return 0 end end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, tensor, firstDerivativeTensor, lowerBoundTensor, upperBoundTensor)

		local collapsedPartialDerivativeTensor = collapseTensor(partialDerivativeTensor, dimensionSizeArray)

		tensor:differentiate{collapsedPartialDerivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.maximum(parameterDictionary)

	local tensorArray = parameterDictionary or {}

	local numberOfTensors = #tensorArray

	local dimensionSizeArrayArray = {}

	local expandedTensorArray = {}

	dimensionSizeArrayArray[1] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[1])

	for i = 2, numberOfTensors, 1 do

		dimensionSizeArrayArray[i] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[i])

		expandedTensorArray[i - 1], expandedTensorArray[i] = AqwamTensorLibrary:broadcast(tensorArray[i - 1], tensorArray[i])

	end

	local resultTensor = AqwamTensorLibrary:applyFunction(math.max, table.unpack(expandedTensorArray))

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then

				local functionToApply = function(derivativeValue, ...)

					local valueArray = {...} 

					local isMaximum = false

					local highestValue = -math.huge

					for j, value in ipairs(valueArray) do

						if (value >= highestValue) then

							isMaximum = (i == j)

							highestValue = value

						end

					end

					return (isMaximum and derivativeValue) or 0

				end

				local currentDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, firstDerivativeTensor, table.unpack(expandedTensorArray))

				local collapsedCurrentDerivativeTensor = collapseTensor(currentDerivativeTensor, dimensionSizeArrayArray[i])

				tensor:differentiate{collapsedCurrentDerivativeTensor}

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, tensorArray})

end

function AHAAutomaticDifferentiationTensor.minimum(parameterDictionary)

	local tensorArray = parameterDictionary or {}

	local numberOfTensors = #tensorArray

	local dimensionSizeArrayArray = {}

	local expandedTensorArray = {}

	dimensionSizeArrayArray[1] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[1])

	for i = 2, numberOfTensors, 1 do

		dimensionSizeArrayArray[i] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[i])

		expandedTensorArray[i - 1], expandedTensorArray[i] = AqwamTensorLibrary:broadcast(tensorArray[i - 1], tensorArray[i])

	end

	local resultTensor = AqwamTensorLibrary:applyFunction(math.min, table.unpack(expandedTensorArray))

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then

				local functionToApply = function(derivativeValue, ...)

					local valueArray = {...} 

					local isMinimum = false

					local lowestValue = -math.huge

					for j, value in ipairs(valueArray) do

						if (value <= lowestValue) then

							isMinimum = (i == j)

							lowestValue = value

						end

					end

					return (isMinimum and derivativeValue) or 0

				end

				local currentDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, firstDerivativeTensor, table.unpack(tensorArray))

				local collapsedCurrentDerivativeTensor = collapseTensor(currentDerivativeTensor, dimensionSizeArrayArray[i])

				tensor:differentiate{collapsedCurrentDerivativeTensor} 

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, tensorArray})

end

function AHAAutomaticDifferentiationTensor:findMaximumValue()

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(selfTensorValue)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local functionToApply = function(value, firstDerivativeValue) return ((value == maximumValue) and firstDerivativeValue) or 0 end

		local chainRuleFirstderivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, tensor, firstDerivativeTensor)

		tensor:differentiate{chainRuleFirstderivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({maximumValue, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:findMinimumValue()

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local minimumValue = AqwamTensorLibrary:findMinimumValue(selfTensorValue)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)
		
		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local functionToApply = function(value, firstDerivativeValue) return ((value == minimumValue) and firstDerivativeValue) or 0 end

		local chainRuleFirstderivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, tensor, firstDerivativeTensor)

		self:differentiate{chainRuleFirstderivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({minimumValue, PartialFirstDerivativeFunction, inputTensorArray})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:__eq(otherTensor)

	return AqwamTensorLibrary:isSameTensor(self, otherTensor)

end

function AHAAutomaticDifferentiationTensor:__add(otherTensor)

	local inputTensorArray = {self, otherTensor}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local otherTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(otherTensor)

	local resultTensor = AqwamTensorLibrary:add(selfTensorValue, otherTensorValue)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local selfTensor = inputTensorArray[1]

		local otherTensor = inputTensorArray[2]

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{selfTensor} then 

			local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selfTensor)

			local collapsedDerivativeTensor = collapseTensor(firstDerivativeTensor, selfDimensionSizeArray)

			selfTensor:differentiate{collapsedDerivativeTensor} 

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{otherTensor} then

			local otherTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(otherTensor)

			local collapsedDerivativeTensor = collapseTensor(firstDerivativeTensor, otherTensorDimensionSizeArray)

			otherTensor:differentiate{collapsedDerivativeTensor}

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:__sub(otherTensor)

	local inputTensorArray = {self, otherTensor}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local otherTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(otherTensor)

	local resultTensor = AqwamTensorLibrary:subtract(selfTensorValue, otherTensorValue)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local selfTensor = inputTensorArray[1]

		local otherTensor = inputTensorArray[2]

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{selfTensor} then

			local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selfTensor)

			local collapsedDerivativeTensor = collapseTensor(firstDerivativeTensor, selfDimensionSizeArray)

			selfTensor:differentiate{collapsedDerivativeTensor}

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{otherTensor} then

			local otherTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(otherTensor)

			local collapsedChainRuleFirstDerivativeTensor = collapseTensor(firstDerivativeTensor, otherTensorDimensionSizeArray)

			otherTensor:differentiate{collapsedChainRuleFirstDerivativeTensor}

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:__mul(otherTensor)

	local inputTensorArray = {self, otherTensor}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local otherTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(otherTensor)

	local resultTensor = AqwamTensorLibrary:multiply(selfTensorValue, otherTensorValue)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local selfTensor = inputTensorArray[1]

		local otherTensor = inputTensorArray[2]

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{selfTensor} then

			local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selfTensor)

			local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(otherTensor, firstDerivativeTensor)

			local collapsedChainRuleFirstDerivativeTensor = collapseTensor(chainRuleFirstDerivativeTensor, selfDimensionSizeArray)

			selfTensor:differentiate{collapsedChainRuleFirstDerivativeTensor}

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{otherTensor} then

			local otherTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(otherTensor)

			local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(selfTensor, firstDerivativeTensor)

			local collapsedChainRuleFirstDerivativeTensor = collapseTensor(chainRuleFirstDerivativeTensor, otherTensorDimensionSizeArray)

			otherTensor:differentiate{collapsedChainRuleFirstDerivativeTensor}

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:__div(otherTensor)

	local inputTensorArray = {self, otherTensor}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local otherTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(otherTensor)

	local resultTensor = AqwamTensorLibrary:divide(selfTensorValue, otherTensorValue)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local selfTensor = inputTensorArray[1]

		local otherTensor = inputTensorArray[2]

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{selfTensor} then

			local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selfTensor)

			local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(otherTensor, firstDerivativeTensor)

			local collapsedChainRuleFirstDerivativeTensor = collapseTensor(chainRuleFirstDerivativeTensor, selfDimensionSizeArray)

			selfTensor:differentiate{collapsedChainRuleFirstDerivativeTensor}

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{otherTensor} then

			local otherTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(otherTensor)

			local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(selfTensor, firstDerivativeTensor)

			local collapsedChainRuleFirstDerivativeTensor = collapseTensor(chainRuleFirstDerivativeTensor, otherTensorDimensionSizeArray)

			otherTensor:differentiate{collapsedChainRuleFirstDerivativeTensor}

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:__unm()

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:unaryMinus(selfTensorValue)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{self} then self:differentiate{AqwamTensorLibrary:unaryMinus(firstDerivativeTensor)} end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, {self}})

end

function AHAAutomaticDifferentiationTensor:__pow(otherTensor)

	local inputTensorArray = {self, otherTensor}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local otherTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(otherTensor)

	local resultTensor = AqwamTensorLibrary:power(selfTensorValue, otherTensorValue)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local selfTensor = inputTensorArray[1]

		local otherTensor = inputTensorArray[2]

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{selfTensor} then

			local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selfTensor)

			local chainRuleFirstDerivativeTensorPart1 = AqwamTensorLibrary:multiply(firstDerivativeTensor, otherTensor)

			local exponentMinusOneTensor = AqwamTensorLibrary:subtract(otherTensor, 1)

			local chainRuleFirstDerivativeTensorPart2 = AqwamTensorLibrary:power(selfTensor, exponentMinusOneTensor)

			local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(chainRuleFirstDerivativeTensorPart1, chainRuleFirstDerivativeTensorPart2)

			local collapsedChainRuleFirstDerivativeTensor = collapseTensor(chainRuleFirstDerivativeTensor, selfDimensionSizeArray)

			selfTensor:differentiate{collapsedChainRuleFirstDerivativeTensor}

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{otherTensor} then

			local otherTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(otherTensor)

			local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(function(base, exponent) return (math.pow(base, exponent) * math.log(base)) end, self, otherTensor)

			local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(partialFirstDerivativeTensor, firstDerivativeTensor)

			local collapsedChainRuleFirstDerivativeTensor = collapseTensor(chainRuleFirstDerivativeTensor, otherTensorDimensionSizeArray)

			otherTensor:differentiate{collapsedChainRuleFirstDerivativeTensor}

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.add(inputTensorArray)

	local resultTensor = AqwamTensorLibrary:add(table.unpack(inputTensorArray))

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		for i, tensor in ipairs(inputTensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				local collapsedDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				tensor:differentiate{collapsedDerivativeTensor}

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.subtract(inputTensorArray)

	local resultTensor = AqwamTensorLibrary:subtract(table.unpack(inputTensorArray))

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		for i, tensor in ipairs(inputTensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				local collapsedDerivativeTensor = collapseTensor(firstDerivativeTensor, dimensionSizeArray)

				tensor:differentiate{collapsedDerivativeTensor}

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.multiply(inputTensorArray)

	local resultTensor = AqwamTensorLibrary:multiply(table.unpack(inputTensorArray))

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		for i, tensor in ipairs(inputTensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then 

				local remainingTensorArray = {}

				for j, tensor in ipairs(inputTensorArray) do

					if (j ~= i) then table.insert(remainingTensorArray, tensor) end

				end

				local currentDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, table.unpack(remainingTensorArray))

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				local collapsedCurrentDerivativeTensor = collapseTensor(currentDerivativeTensor, dimensionSizeArray)

				tensor:differentiate{collapsedCurrentDerivativeTensor}

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.divide(inputTensorArray)

	local resultTensor = AqwamTensorLibrary:divide(table.unpack(inputTensorArray))

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		for i, tensor in ipairs(inputTensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then 

				local remainingTensorArray = {}

				for j, tensor in ipairs(inputTensorArray) do

					if (j ~= i) then table.insert(remainingTensorArray, tensor) end

				end

				local currentDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, table.unpack(remainingTensorArray))

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				local collapsedCurrentDerivativeTensor = collapseTensor(currentDerivativeTensor, dimensionSizeArray)

				tensor:differentiate{collapsedCurrentDerivativeTensor}

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:sum(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local dimension = parameterDictionary.dimension or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:sum(selfTensorValue, dimension)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then return end

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

		if (dimension) then

			firstDerivativeTensor = AqwamTensorLibrary:expandDimensionSizes(firstDerivativeTensor, dimensionSizeArray)

		else

			firstDerivativeTensor = AqwamTensorLibrary:expandNumberOfDimensions(firstDerivativeTensor, dimensionSizeArray)

		end

		tensor:differentiate{firstDerivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:unaryMinus()

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:unaryMinus(selfTensorValue)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then tensor:differentiate{AqwamTensorLibrary:unaryMinus(firstDerivativeTensor)} end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:power(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local otherTensor = parameterDictionary.otherTensor or parameterDictionary[1]

	local inputTensorArray = {self, otherTensor}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local otherTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(otherTensor)

	local resultTensor = AqwamTensorLibrary:power(selfTensorValue, otherTensorValue)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local selfTensor = inputTensorArray[1]

		local otherTensor = inputTensorArray[2]

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{selfTensor} then 

			local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

			local chainRuleFirstDerivativeTensorPart1 = AqwamTensorLibrary:multiply(firstDerivativeTensor, otherTensor)

			local exponentMinusOneTensor = AqwamTensorLibrary:subtract(otherTensor, 1)

			local chainRuleFirstDerivativeTensorPart2 = AqwamTensorLibrary:power(selfTensor, exponentMinusOneTensor)

			local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(chainRuleFirstDerivativeTensorPart1, chainRuleFirstDerivativeTensorPart2)

			local collapsedChainRuleFirstDerivativeTensor = collapseTensor(chainRuleFirstDerivativeTensor, selfDimensionSizeArray)

			selfTensor:differentiate{collapsedChainRuleFirstDerivativeTensor}

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{otherTensor} then

			local otherTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(otherTensor)

			local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(function(base, exponent) return (math.pow(base, exponent) * math.log(base)) end, self, otherTensor)

			local collapsedChainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(partialFirstDerivativeTensor, firstDerivativeTensor)

			local collapsedChainRuleDerivativeTensor = collapseTensor(collapsedChainRuleFirstDerivativeTensor, otherTensorDimensionSizeArray)

			otherTensor:differentiate{collapsedChainRuleFirstDerivativeTensor}

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:dotProduct(parameterDictionary) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local otherTensor = parameterDictionary.otherTensor or parameterDictionary[1]

	local inputTensorArray = {self, otherTensor}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local otherTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(otherTensor)

	local resultTensor = AqwamTensorLibrary:dotProduct(selfTensorValue, otherTensorValue)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local selfTensor = inputTensorArray[1]

		local otherTensor = inputTensorArray[2]

		local selfTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selfTensor)

		local otherTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(otherTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{selfTensor} then

			local otherTensorNumberOfDimensions = #otherTensorDimensionSizeArray

			local transposedOther = AqwamTensorLibrary:transpose(otherTensor, {otherTensorNumberOfDimensions - 1, otherTensorNumberOfDimensions})

			local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(firstDerivativeTensor, transposedOther)

			local collapsedChainRuleFirstDerivativeTensor = collapseTensor(chainRuleFirstDerivativeTensor, selfTensorDimensionSizeArray)

			selfTensor:differentiate{collapsedChainRuleFirstDerivativeTensor} 

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{otherTensor} then

			local selfNumberOfDimensions = #selfTensorDimensionSizeArray

			local transposedSelf = AqwamTensorLibrary:transpose(selfTensor, {selfNumberOfDimensions - 1, selfNumberOfDimensions})

			local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(transposedSelf, firstDerivativeTensor)

			local collapsedChainRuleFirstDerivativeTensor = collapseTensor(chainRuleFirstDerivativeTensor, otherTensorDimensionSizeArray)

			otherTensor:differentiate{collapsedChainRuleFirstDerivativeTensor} 

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:extract(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local originDimensionIndexArray = parameterDictionary.originDimensionIndexArray or parameterDictionary[1]

	local targetDimensionIndexArray = parameterDictionary.targetDimensionIndexArray or parameterDictionary[2]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:extract(selfTensorValue, originDimensionIndexArray, targetDimensionIndexArray)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local originalTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

		local originDimensionIndexArray = originDimensionIndexArray

		local targetDimensionIndexArray = targetDimensionIndexArray

		local numberOfDimensions = #originalTensorDimensionSizeArray

		local headPaddingDimensionSizeArray = {}

		local tailPaddingDimensionSizeArray = {}

		for dimension = 1, numberOfDimensions, 1 do

			headPaddingDimensionSizeArray[dimension] = originDimensionIndexArray[dimension] - 1

			tailPaddingDimensionSizeArray[dimension] = originalTensorDimensionSizeArray[dimension] - targetDimensionIndexArray[dimension]

		end

		for dimension = numberOfDimensions, 1, -1 do

			local firstDerivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor)

			local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

			local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

			if (headPaddingDimensionSize >= 1) then

				local tensorHeadPaddingDimensionSizeArray = table.clone(firstDerivativeTensorDimensionSizeArray)

				tensorHeadPaddingDimensionSizeArray[dimension] = headPaddingDimensionSize

				local headPaddingTensor = AqwamTensorLibrary:createTensor(tensorHeadPaddingDimensionSizeArray)

				firstDerivativeTensor = AqwamTensorLibrary:concatenate(headPaddingTensor, firstDerivativeTensor, dimension)

			end

			if (tailPaddingDimensionSize >= 1) then

				local tensorTailPaddingDimensionSizeArray = table.clone(firstDerivativeTensorDimensionSizeArray)

				tensorTailPaddingDimensionSizeArray[dimension] = tailPaddingDimensionSize

				local tailPaddingTensor = AqwamTensorLibrary:createTensor(tensorTailPaddingDimensionSizeArray)

				firstDerivativeTensor = AqwamTensorLibrary:concatenate(firstDerivativeTensor, tailPaddingTensor, dimension)

			end

		end

		tensor:differentiate{firstDerivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.concatenate(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensorArray = parameterDictionary.tensorArray or parameterDictionary

	local numberOfArguments = #tensorArray

	local dimensionIndex = tensorArray[numberOfArguments]

	if (type(dimensionIndex) ~= "number") then error("The final argument must be a number in order for it to be used as dimension index.") end

	table.remove(tensorArray, numberOfArguments)

	local resultTensor

	for i, tensor in ipairs(tensorArray) do

		if (i > 1) then

			resultTensor = AqwamTensorLibrary:concatenate(resultTensor, tensor, dimensionIndex)

		else

			resultTensor = tensor

		end

	end

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local extractedDerivativeTensorArray = {}

		local firstDerivativeTensorDimensionArray = AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor)

		local originDimensionIndexArray = table.create(#firstDerivativeTensorDimensionArray, 1)

		local targetDimensionIndexArray = table.clone(firstDerivativeTensorDimensionArray)

		targetDimensionIndexArray[dimensionIndex] = 0

		for _, tensor in ipairs(tensorArray) do

			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

			targetDimensionIndexArray[dimensionIndex] = originDimensionIndexArray[dimensionIndex] + dimensionSizeArray[dimensionIndex] - 1

			local extractedDerivativeTensor = AqwamTensorLibrary:extract(firstDerivativeTensor, originDimensionIndexArray, targetDimensionIndexArray)

			originDimensionIndexArray[dimensionIndex] = originDimensionIndexArray[dimensionIndex] + dimensionSizeArray[dimensionIndex]

			table.insert(extractedDerivativeTensorArray, extractedDerivativeTensor)

		end

		for i, tensor in ipairs(tensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then tensor:differentiate{extractedDerivativeTensorArray[i]} end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, tensorArray})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:transpose(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local dimensionArray = parameterDictionary.dimensionArray or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:transpose(selfTensorValue, dimensionArray)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then tensor:differentiate(AqwamTensorLibrary:transpose(firstDerivativeTensor, dimensionArray)) end

	end

	return self.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:flatten(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local dimensionArray = parameterDictionary.dimensionArray or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:flatten(selfTensorValue, dimensionArray)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

		firstDerivativeTensor = AqwamTensorLibrary:reshape(firstDerivativeTensor, dimensionSizeArray)

		tensor:differentiate{firstDerivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:reshape(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:reshape(selfTensorValue, dimensionSizeArray)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local originalDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

		firstDerivativeTensor = AqwamTensorLibrary:reshape(firstDerivativeTensor, originalDimensionSizeArray)

		tensor:differentiate{firstDerivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:permute(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local dimensionArray = parameterDictionary.dimensionArray or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:permute(selfTensorValue, dimensionArray)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local originalDimensionArray = createOriginalDimensionArray(dimensionArray)

		firstDerivativeTensor = AqwamTensorLibrary:permute(firstDerivativeTensor, originalDimensionArray)

		tensor:differentiate{firstDerivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:mean(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local dimension = parameterDictionary.dimension or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:mean(selfTensorValue, dimension)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local dimensionSize = AqwamTensorLibrary:getDimensionSizeArray(tensor)[dimension]

		firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, dimensionSize)

		self:differentiate{firstDerivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})	

end

function AHAAutomaticDifferentiationTensor:standardDeviation(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local dimension = parameterDictionary.dimension or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:standardDeviation(selfTensorValue, dimension)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local dimensionSize = AqwamTensorLibrary:getDimensionSizeArray(tensor)[dimension]

		local chainRuleFirstDerivativeTensorPart1 = AqwamTensorLibrary:multiply(2, resultTensor, dimensionSize)

		firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, chainRuleFirstDerivativeTensorPart1)

		tensor:differentiate{firstDerivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})	

end

function AHAAutomaticDifferentiationTensor:zScoreNormalization(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local dimension = parameterDictionary.dimension or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:zScoreNormalization(selfTensorValue, dimension)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local standardDeviationTensor = AqwamTensorLibrary:standardDeviation(tensor, dimension)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, standardDeviationTensor)

		tensor:differentiate{chainRuleFirstDerivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})	

end

function AHAAutomaticDifferentiationTensor:absolute()

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local functionToApply = function (value) return (((value >= 0) and value) or -value) end

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:applyFunction(functionToApply, selfTensorValue)

	local inputTensorArray = {self}

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local functionToApply = function (firstDerivativeValue, value) return (((value >= 0) and firstDerivativeValue) or -firstDerivativeValue) end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, firstDerivativeTensor, tensor)

		tensor:differentiate{chainRuleFirstDerivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})	

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:expandDimensionSizes(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local targetDimensionSizeArray = parameterDictionary.targetDimensionSizeArray or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:expandDimensionSizes(selfTensorValue, targetDimensionSizeArray)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

		local chainRuleFirstDerivativeTensor = firstDerivativeTensor

		for dimension, dimensionSize in ipairs(tensorDimensionSizeArray) do

			if (dimensionSize == 1) and (targetDimensionSizeArray[dimension] > 1) then

				chainRuleFirstDerivativeTensor = AqwamTensorLibrary:sum(chainRuleFirstDerivativeTensor, dimension)

			end

		end

		tensor:differentiate{chainRuleFirstDerivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:expandNumberOfDimensions(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local dimensionSizeToAddArray = parameterDictionary.dimensionSizeToAddArray or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local resultTensor = AqwamTensorLibrary:expandNumberOfDimensions(selfTensorValue, dimensionSizeToAddArray)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		local tensor = inputTensorArray[1]

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end

		local numberOfDimensionsToSum = #dimensionSizeToAddArray

		local chainRuleFirstDerivativeTensor = firstDerivativeTensor

		for i = 1, numberOfDimensionsToSum, 1 do chainRuleFirstDerivativeTensor = AqwamTensorLibrary:sum(chainRuleFirstDerivativeTensor, 1)[1] end -- Remove the first dimension as it is redundant and does not carry any values. If it is not removed, this tensor might broadcast its dimension size elsewhere like during the gradient descent.

		tensor:differentiate{chainRuleFirstDerivativeTensor}

	end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor.createTensor(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local allValues = parameterDictionary.allValues or parameterDictionary[2]

	local tensor = AqwamTensorLibrary:createTensor(dimensionSizeArray, allValues)

	return AHAAutomaticDifferentiationTensor.new({tensor})

end

function AHAAutomaticDifferentiationTensor.createRandomNormalTensor(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local mean = parameterDictionary.mean or parameterDictionary[2]

	local standardDeviation = parameterDictionary.standardDeviation or parameterDictionary[3]

	local tensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray, mean, standardDeviation)

	return AHAAutomaticDifferentiationTensor.new({tensor})

end

function AHAAutomaticDifferentiationTensor.createRandomUniformTensor(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local minimumValue = parameterDictionary.minimumValue or parameterDictionary[2]

	local maximumValue = parameterDictionary.maximumValue or parameterDictionary[3]

	local tensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)

	return AHAAutomaticDifferentiationTensor.new({tensor})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:isAutomaticDifferentiationTensor()

	return true

end

function AHAAutomaticDifferentiationTensor:differentiate(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local firstDerivativeTensor = parameterDictionary.firstDerivativeTensor or parameterDictionary[1]

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue(self)

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selfTensorValue)

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	if (not firstDerivativeTensor) then

		if (tensorNumberOfDimensions >= 1) then

			firstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray, 1)

		else

			firstDerivativeTensor = 1

		end

	else

		-- if (#firstDerivativeTensor == 0) then firstDerivativeTensor = 0 end -- Our __index function could not return the pure scalar value due to being wrapped around the automatic differentiation tensor table. So this was added to prevent a bug where the first derivative tensor has 0 dimensions when the original tensor has 1 dimension.

		local firstDerivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor)

		local firstDerivativeTensorNumberOfDimensions = #firstDerivativeTensorDimensionSizeArray
		
		local isInputScalar = (tensorNumberOfDimensions == 0) 
		
		local isFirstDerivativeScalar = (firstDerivativeTensorNumberOfDimensions == 0)
		
		if (not isInputScalar) and (isFirstDerivativeScalar) then
			
			error("Unable to differentiate. The scalar derivative cannot be applied to the original tensor.")
			
		end
		
		if (not isInputScalar) and (firstDerivativeTensorNumberOfDimensions ~= tensorNumberOfDimensions) then
			
			error("Unable to differentiate. The derivative tensor has " .. firstDerivativeTensorNumberOfDimensions .. " dimensions, but the original tensor has " .. tensorNumberOfDimensions .. ".")
			
		end

		if (not isInputScalar) then
			
			for dimension, firstDerivativeTensorDimensionSize in ipairs(firstDerivativeTensorDimensionSizeArray) do

				local tensorDimensionSize = tensorDimensionSizeArray[dimension]

				if (firstDerivativeTensorDimensionSize ~= tensorDimensionSize) then
					
					error("Unable to differentiate. The derivative tensor has a dimension size of " .. firstDerivativeTensorDimensionSize .. " at dimension " .. dimension .. ", but the original tensor has " .. tensorDimensionSize .. ".")
					
				end

			end
			
		end

	end

	local PartialFirstDerivativeFunction = self.PartialFirstDerivativeFunction

	if (PartialFirstDerivativeFunction) then PartialFirstDerivativeFunction(firstDerivativeTensor) end

	local totalFirstDerivativeTensor = self.totalFirstDerivativeTensor

	if (not totalFirstDerivativeTensor) then

		totalFirstDerivativeTensor = firstDerivativeTensor

	else

		totalFirstDerivativeTensor = AqwamTensorLibrary:add(totalFirstDerivativeTensor, firstDerivativeTensor)

	end

	self.totalFirstDerivativeTensor = totalFirstDerivativeTensor

end

function AHAAutomaticDifferentiationTensor:copy()

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	return deepCopyTable(self)

end

function AHAAutomaticDifferentiationTensor:getTensor(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[1]

	if (doNotDeepCopy) then

		return self.tensor

	else

		return deepCopyTable(self.tensor)

	end

end

function AHAAutomaticDifferentiationTensor:setTensor(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[2]

	if (doNotDeepCopy) then

		self.tensor = tensor

	else

		self.tensor = deepCopyTable(tensor)

	end

end

function AHAAutomaticDifferentiationTensor:getTotalFirstDerivativeTensor(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[1]

	if (doNotDeepCopy) then 

		return self.totalFirstDerivativeTensor

	else

		return deepCopyTable(self.totalFirstDerivativeTensor)

	end

end

function AHAAutomaticDifferentiationTensor:setTotalFirstDerivativeTensor(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local totalFirstDerivativeTensor = parameterDictionary.totalFirstDerivativeTensor or parameterDictionary[1]

	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[2]

	if (doNotDeepCopy) then

		self.totalFirstDerivativeTensor = totalFirstDerivativeTensor

	else

		self.totalFirstDerivativeTensor = deepCopyTable(totalFirstDerivativeTensor)

	end

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:__tostring()

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local tensor = self.tensor

	if (type(tensor) == "table") then

		return AqwamTensorLibrary:generateTensorString(tensor)

	else

		return tostring(tensor)	

	end

end

function AHAAutomaticDifferentiationTensor:__len()

	local tensor = self.tensor

	if (type(tensor) == "table") then

		return #tensor 

	else

		return 0

	end

end

function AHAAutomaticDifferentiationTensor:__index(index)

	if (type(index) == "number") then

		local tensor = self.tensor

		if (type(tensor) == "table") then

			return rawget(tensor, index)

		else

			return nil

		end

	else

		return rawget(AHAAutomaticDifferentiationTensor, index)

	end

end

function AHAAutomaticDifferentiationTensor:__newindex(index, value)

	rawset(self, index, value)

end

function AHAAutomaticDifferentiationTensor:destroy(parameterDictionary)

	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local areDescendantsDestroyed = parameterDictionary.areDescendantsDestroyed or parameterDictionary[1]

	local destroyFirstInputTensor = parameterDictionary.destroyFirstInputTensor or parameterDictionary[2]

	local inputTensorArray = self.inputTensorArray

	if (areDescendantsDestroyed) and (inputTensorArray) then

		for _, tensor in ipairs(inputTensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then

				if (tensor.inputTensorArray) or (destroyFirstInputTensor) then

					tensor:destroy{areDescendantsDestroyed, destroyFirstInputTensor}

				end

			end

		end

	end

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return AHAAutomaticDifferentiationTensor