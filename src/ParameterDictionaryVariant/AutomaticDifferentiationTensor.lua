--[[

	--------------------------------------------------------------------

	Aqwam's Deep Learning Library (DataPredict Torch)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict-Neural/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require(script.Parent.AqwamTensorLibraryLinker.Value)

local AHAAutomaticDifferentiationTensor = {}

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

function AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)

	local isAutomaticDifferentiationTensor = pcall(function()

		tensor:isAutomaticDifferentiationTensor()

	end)

	return isAutomaticDifferentiationTensor

end

function AHAAutomaticDifferentiationTensor:collapseTensor(tensor, targetDimensionSizeArray)

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

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor.new(parameterDictionary)

	local self = setmetatable({}, AHAAutomaticDifferentiationTensor)

	self.tensor = parameterDictionary.tensor or parameterDictionary[1]

	self.PartialDerivativeFunction = parameterDictionary.PartialDerivativeFunction or parameterDictionary[2]

	self.inputTensorArray = parameterDictionary.inputTensorArray or parameterDictionary[3]

	self.totalDerivativeTensor = nil

	return self

end

function AHAAutomaticDifferentiationTensor.radian(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local result = AqwamTensorLibrary:applyFunction(math.rad, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local radiansPerDegree = math.pi / 180

		tensor:differentiate(AqwamTensorLibrary:multiply(radiansPerDegree, derivativeTensor))

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {tensor}})

end

function AHAAutomaticDifferentiationTensor.degree(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local result = AqwamTensorLibrary:applyFunction(math.deg, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local degreesPerRadian = 180 / math.pi

		tensor:differentiate(AqwamTensorLibrary:multiply(degreesPerRadian, derivativeTensor))

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {tensor}})

end

function AHAAutomaticDifferentiationTensor.sin(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local result = AqwamTensorLibrary:applyFunction(math.sin, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(math.cos, tensor)

		tensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {tensor}})

end

function AHAAutomaticDifferentiationTensor.cos(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local result = AqwamTensorLibrary:applyFunction(math.cos, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local partialDerivativeFunctionToApply = function (radian) return -math.sin(radian) end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {tensor}})

end

function AHAAutomaticDifferentiationTensor.tan(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local result = AqwamTensorLibrary:applyFunction(math.tan, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local partialDerivativeFunctionToApply = function (radian) return math.pow((1 / math.cos(radian)), 2) end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {tensor}})

end

function AHAAutomaticDifferentiationTensor.exponent(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local result = AqwamTensorLibrary:applyFunction(math.exp, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		tensor:differentiate(AqwamTensorLibrary:multiply(result, derivativeTensor))

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {tensor}})

end

function AHAAutomaticDifferentiationTensor.logarithm(parameterDictionary)

	local numberTensor = parameterDictionary.numberTensor or parameterDictionary[1]

	local baseTensor = parameterDictionary.baseTensor or parameterDictionary[2]

	local result = AqwamTensorLibrary:applyFunction(math.log, numberTensor, baseTensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(numberTensor) then

			local numberTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(numberTensor)

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, numberTensorDimensionSizeArray)

			local partialDerivativeTensor

			if (baseTensor) then

				local partialDerivativeFunctionToApply = function (number, base) return (1 / (number * math.log(base))) end

				partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, numberTensor, baseTensor)

			else

				local partialDerivativeFunctionToApply = function (number) return (1 / number) end

				partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, numberTensor)

			end

			numberTensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)) 

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(baseTensor) then

			local baseTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(baseTensor)

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, baseTensorDimensionSizeArray)

			local partialDerivativeFunctionToApply = function (number, base) return -(math.log(number) / (base * math.pow(math.log(base), 2))) end

			local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, numberTensor, baseTensorDimensionSizeArray)

			baseTensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)) 

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {numberTensor, baseTensor}})

end

function AHAAutomaticDifferentiationTensor.clamp(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local lowerBoundTensor = parameterDictionary.lowerBoundTensor or parameterDictionary[2]

	local upperBoundTensor = parameterDictionary.upperBoundTensor or parameterDictionary[3]

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	lowerBoundTensor = lowerBoundTensor or -math.huge

	upperBoundTensor = upperBoundTensor or math.huge

	local result = AqwamTensorLibrary:applyfunction(math.clamp, tensor, lowerBoundTensor, upperBoundTensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local functionToApply = function(value, derivative, lowerBoundValue, upperBoundValue) if ((value >= lowerBoundValue) and (value <= upperBoundValue)) then return value else return 0 end end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, tensor, derivativeTensor, lowerBoundTensor, upperBoundTensor)

		local collapsedPartialDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(partialDerivativeTensor, dimensionSizeArray)

		tensor:differentiate(collapsedPartialDerivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {tensor, lowerBoundTensor, upperBoundTensor}})

end

function AHAAutomaticDifferentiationTensor.maximum(parameterDictionary)

	local tensorArray = parameterDictionary.tensorArray or parameterDictionary[1]

	local numberOfTensors = #tensorArray

	local dimensionSizeArrayArray = {}

	local expandedTensorArray = {}

	dimensionSizeArrayArray[1] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[1])

	for i = 1, (numberOfTensors - 1), 1 do

		dimensionSizeArrayArray[i] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[i + 1])

		expandedTensorArray[i], expandedTensorArray[i + 1] = AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensorArray[i], tensorArray[i + 1])

	end

	local result = AqwamTensorLibrary:applyfunction(math.max, table.unpack(tensorArray))

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor) then

				local functionToApply = function(derivativeValue, ...)

					local isMaximum = false

					local highestValue = -math.huge

					for j, value in ipairs(...) do

						if (value >= highestValue) then

							isMaximum = (i == j)

							highestValue = value

						end

					end

					return (isMaximum and derivativeValue) or 0

				end

				local currentDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, derivativeTensor, table.unpack(tensorArray))

				local collapsedCurrentDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(currentDerivativeTensor, dimensionSizeArrayArray[i])

				tensor:differentiate(collapsedCurrentDerivativeTensor) 

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, tensorArray})

end

function AHAAutomaticDifferentiationTensor.minimum(parameterDictionary)

	local tensorArray = parameterDictionary.tensorArray or parameterDictionary[1]

	local numberOfTensors = #tensorArray

	local dimensionSizeArrayArray = {}

	local expandedTensorArray = {}

	dimensionSizeArrayArray[1] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[1])

	for i = 1, (numberOfTensors - 1), 1 do

		dimensionSizeArrayArray[i] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[i + 1])

		expandedTensorArray[i], expandedTensorArray[i + 1] = AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensorArray[i], tensorArray[i + 1])

	end

	local result = AqwamTensorLibrary:applyfunction(math.min, table.unpack(tensorArray))

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor) then

				local functionToApply = function(derivativeValue, ...)

					local isMinimum = false

					local lowestValue = -math.huge

					for j, value in ipairs(...) do

						if (value <= lowestValue) then

							isMinimum = (i == j)

							lowestValue = value

						end

					end

					return (isMinimum and derivativeValue) or 0

				end

				local currentDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, derivativeTensor, table.unpack(tensorArray))

				local collapsedCurrentDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(currentDerivativeTensor, dimensionSizeArrayArray[i])

				tensor:differentiate(collapsedCurrentDerivativeTensor) 

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, tensorArray})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:__eq(other)

	return AqwamTensorLibrary:isSameTensor(self, other)

end

function AHAAutomaticDifferentiationTensor:__add(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:add(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(collapsedDerivativeTensor) 

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, otherDimensionSizeArray)

			other:differentiate(collapsedDerivativeTensor) 

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self, other}})

end

function AHAAutomaticDifferentiationTensor:__sub(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:subtract(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(collapsedDerivativeTensor) 

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, otherDimensionSizeArray)

			other:differentiate(collapsedDerivativeTensor) 

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self, other}})

end

function AHAAutomaticDifferentiationTensor:__mul(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:multiply(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(AqwamTensorLibrary:multiply(other, collapsedDerivativeTensor))

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, otherDimensionSizeArray)

			other:differentiate(AqwamTensorLibrary:multiply(self, collapsedDerivativeTensor))

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self, other}})

end

function AHAAutomaticDifferentiationTensor:__div(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:divide(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(AqwamTensorLibrary:multiply(other, collapsedDerivativeTensor))

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, otherDimensionSizeArray)

			other:differentiate(AqwamTensorLibrary:multiply(self, collapsedDerivativeTensor))

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self, other}})

end

function AHAAutomaticDifferentiationTensor:__unm()

	local result = AqwamTensorLibrary:unaryMinus(self)

	local PartialDerivativeFunction = function(derivativeTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:unaryMinus(derivativeTensor)) end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self}})

end

function AHAAutomaticDifferentiationTensor:__pow(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:power(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(AqwamTensorLibrary:multiply(other, collapsedDerivativeTensor)) 

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(other) then 

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, otherDimensionSizeArray)

			local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(function(base, exponent) return (math.pow(base, exponent) * math.log(base)) end, self, other)

			other:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)) 

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self, other}})

end

function AHAAutomaticDifferentiationTensor:add(parameterDictionary)

	local tensorArray = parameterDictionary.tensorArray or parameterDictionary[1]

	local result = AqwamTensorLibrary:add(table.unpack(tensorArray))

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor) then

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, dimensionSizeArray)

				tensor:differentiate(collapsedDerivativeTensor) 

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, tensorArray})

end

function AHAAutomaticDifferentiationTensor:subtract(parameterDictionary)

	local tensorArray = parameterDictionary.tensorArray or parameterDictionary[1]

	local result = AqwamTensorLibrary:subtract(table.unpack(tensorArray))

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor) then

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, dimensionSizeArray)

				tensor:differentiate(collapsedDerivativeTensor) 

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, tensorArray})

end

function AHAAutomaticDifferentiationTensor:multiply(parameterDictionary)

	local tensorArray = parameterDictionary.tensorArray or parameterDictionary[1]

	local result = AqwamTensorLibrary:multiply(table.unpack(tensorArray))

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor) then 

				local remainingTensorArray = {}

				for j, tensor in ipairs(tensorArray) do

					if (j ~= i) then table.insert(remainingTensorArray, tensor) end

				end

				local currentDerivativeTensor = AqwamTensorLibrary:multiply(derivativeTensor, table.unpack(remainingTensorArray))

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				local collapsedCurrentDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(currentDerivativeTensor, dimensionSizeArray)

				tensor:differentiate(collapsedCurrentDerivativeTensor)

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, tensorArray})

end

function AHAAutomaticDifferentiationTensor:divide(parameterDictionary)

	local tensorArray = parameterDictionary.tensorArray or parameterDictionary[1]

	local result = AqwamTensorLibrary:divide(table.unpack(tensorArray))

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor) then 

				local remainingTensorArray = {}

				for j, tensor in ipairs(tensorArray) do

					if (j ~= i) then table.insert(remainingTensorArray, tensor) end

				end

				local currentDerivativeTensor = AqwamTensorLibrary:multiply(derivativeTensor, table.unpack(remainingTensorArray))

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				local collapsedCurrentDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(currentDerivativeTensor, dimensionSizeArray)

				tensor:differentiate(collapsedCurrentDerivativeTensor)

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, tensorArray})

end

function AHAAutomaticDifferentiationTensor:sum(parameterDictionary)

	local dimension = parameterDictionary.dimension or parameterDictionary[1]

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local result = AqwamTensorLibrary:sum(self, dimension)

	local PartialDerivativeFunction = function(derivativeTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self) then 

			if (dimension) then

				derivativeTensor = AqwamTensorLibrary:expandDimensionSizes(derivativeTensor, dimensionSizeArray)

			else

				derivativeTensor = AqwamTensorLibrary:expandNumberOfDimensions(derivativeTensor, dimensionSizeArray)

			end

			self:differentiate(derivativeTensor) 

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self}})

end

function AHAAutomaticDifferentiationTensor:unaryMinus()

	local result = AqwamTensorLibrary:unaryMinus(self)

	local PartialDerivativeFunction = function(derivativeTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:unaryMinus(derivativeTensor)) end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self}})

end

function AHAAutomaticDifferentiationTensor:power(parameterDictionary)
	
	local other = parameterDictionary.other or parameterDictionary[1]

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:power(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(AqwamTensorLibrary:multiply(other, collapsedDerivativeTensor)) 

		end

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(other) then 

			local collapsedDerivativeTensor = AHAAutomaticDifferentiationTensor:collapseTensor(derivativeTensor, otherDimensionSizeArray)

			local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(function(base, exponent) return (math.pow(base, exponent) * math.log(base)) end, self, other)

			other:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)) 

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self, other}})

end

function AHAAutomaticDifferentiationTensor:dotProduct(parameterDictionary) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc
	
	local other = parameterDictionary.other or parameterDictionary[1]
	
	local result = AqwamTensorLibrary:dotProduct(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		local otherNumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(other)
		local selfNumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(self)

		local transposedOther = AqwamTensorLibrary:transpose(other, {otherNumberOfDimensions - 1, otherNumberOfDimensions})
		local transposedSelf = AqwamTensorLibrary:transpose(self, {selfNumberOfDimensions - 1, selfNumberOfDimensions})

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:dotProduct(derivativeTensor, transposedOther)) end
		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(other) then other:differentiate(AqwamTensorLibrary:dotProduct(transposedSelf, derivativeTensor)) end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self, other}})

end

function AHAAutomaticDifferentiationTensor:extract(parameterDictionary)
	
	local originDimensionIndexArray = parameterDictionary.originDimensionIndexArray or parameterDictionary[1]
	
	local targetDimensionIndexArray = parameterDictionary.targetDimensionIndexArray or parameterDictionary[2]

	local originalTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local result = AqwamTensorLibrary:extract(self, originDimensionIndexArray, targetDimensionIndexArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self)) then return end

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

			local derivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(derivativeTensor)

			local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

			local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

			if (headPaddingDimensionSize >= 1) then

				local tensorHeadPaddingDimensionSizeArray = table.clone(derivativeTensorDimensionSizeArray)

				tensorHeadPaddingDimensionSizeArray[dimension] = headPaddingDimensionSize

				local headPaddingTensor = AqwamTensorLibrary:createTensor(tensorHeadPaddingDimensionSizeArray)

				derivativeTensor = AqwamTensorLibrary:concatenate(headPaddingTensor, derivativeTensor, dimension)

			end

			if (tailPaddingDimensionSize >= 1) then

				local tensorTailPaddingDimensionSizeArray = table.clone(derivativeTensorDimensionSizeArray)

				tensorTailPaddingDimensionSizeArray[dimension] = tailPaddingDimensionSize

				local tailPaddingTensor = AqwamTensorLibrary:createTensor(tensorTailPaddingDimensionSizeArray)

				derivativeTensor = AqwamTensorLibrary:concatenate(derivativeTensor, tailPaddingTensor, dimension)

			end

		end

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self}})

end

function AHAAutomaticDifferentiationTensor.concatenate(parameterDictionary)

	local tensorArray = parameterDictionary.tensorArray or parameterDictionary[1]

	local numberOfArguments = #tensorArray

	local dimensionIndex = tensorArray[numberOfArguments]

	if (type(dimensionIndex) ~= "number") then error("The final argument must be a number in order for it to be used as dimension index.") end

	table.remove(tensorArray, numberOfArguments)

	local result

	for i, tensor in ipairs(tensorArray) do

		if (i > 1) then

			result = AqwamTensorLibrary:concatenate(result, tensor, dimensionIndex)

		else

			result = tensor

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		local extractedDerivativeTensorArray = {}

		local derivativeTensorDimensionArray = AqwamTensorLibrary:getDimensionSizeArray(derivativeTensor)

		local originDimensionIndexArray = table.create(#derivativeTensorDimensionArray, 1)

		local targetDimensionIndexArray = table.clone(derivativeTensorDimensionArray)

		targetDimensionIndexArray[dimensionIndex] = 0

		for _, tensor in ipairs(tensorArray) do

			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

			targetDimensionIndexArray[dimensionIndex] = originDimensionIndexArray[dimensionIndex] + dimensionSizeArray[dimensionIndex] - 1

			local extractedDerivativeTensor = AqwamTensorLibrary:extract(derivativeTensor, originDimensionIndexArray, targetDimensionIndexArray)

			originDimensionIndexArray[dimensionIndex] = originDimensionIndexArray[dimensionIndex] + dimensionSizeArray[dimensionIndex]

			table.insert(extractedDerivativeTensorArray, extractedDerivativeTensor)

		end

		for i, tensor in ipairs(tensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor) then tensor:differentiate(extractedDerivativeTensorArray[i]) end

		end

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, tensorArray})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:transpose(parameterDictionary)
	
	local dimensionIndexArray = parameterDictionary.dimensionIndexArray or parameterDictionary[1]

	local result = AqwamTensorLibrary:transpose(self, dimensionIndexArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:transpose(derivativeTensor, dimensionIndexArray)) end

	end

	return self.new({result, PartialDerivativeFunction, {self}})

end

function AHAAutomaticDifferentiationTensor:flatten(parameterDictionary)
	
	local dimensionArray = parameterDictionary.dimensionArray or parameterDictionary[1]

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local result = AqwamTensorLibrary:flatten(self, dimensionArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:reshape(derivativeTensor, dimensionSizeArray)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self}})

end

function AHAAutomaticDifferentiationTensor:reshape(parameterDictionary)
	
	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local originalDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local result = AqwamTensorLibrary:reshape(self, dimensionSizeArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:reshape(derivativeTensor, originalDimensionSizeArray)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self}})

end

function AHAAutomaticDifferentiationTensor:permute(parameterDictionary)
	
	local dimensionArray = parameterDictionary.dimensionArray or parameterDictionary[1]

	local originalDimensionArray = createOriginalDimensionArray(dimensionArray)

	local result = AqwamTensorLibrary:permute(self, dimensionArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:permute(derivativeTensor, originalDimensionArray)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self}})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:mean(parameterDictionary)
	
	local dimension = parameterDictionary.dimension or parameterDictionary[1]

	local result = AqwamTensorLibrary:mean(self, dimension)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self)) then return end

		local dimensionSize = AqwamTensorLibrary:getDimensionSizeArray(self)[dimension]

		derivativeTensor = AqwamTensorLibrary:divide(derivativeTensor, dimensionSize)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self}})	

end

function AHAAutomaticDifferentiationTensor:standardDeviation(parameterDictionary)
	
	local dimension = parameterDictionary.dimension or parameterDictionary[1]

	local result = AqwamTensorLibrary:standardDeviation(self, dimension)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self)) then return end

		local dimensionSize = AqwamTensorLibrary:getDimensionSizeArray(self)[dimension]

		local chainRuleFirstDerivativeTensorPart1 = AqwamTensorLibrary:multiply(2, result, dimensionSize)

		derivativeTensor = AqwamTensorLibrary:divide(derivativeTensor, chainRuleFirstDerivativeTensorPart1)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self}})	

end

function AHAAutomaticDifferentiationTensor:zScoreNormalization(parameterDictionary)
	
	local dimension = parameterDictionary.dimension or parameterDictionary[1]

	local result = AqwamTensorLibrary:standardDeviation(self, dimension)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self)) then return end

		local standardDeviationTensor = AqwamTensorLibrary:standardDeviation(self, dimension)

		derivativeTensor = AqwamTensorLibrary:divide(derivativeTensor, standardDeviationTensor)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new({result, PartialDerivativeFunction, {self}})	

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor.createTensor(parameterDictionary)

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local allValues = parameterDictionary.allValues or parameterDictionary[2]

	local tensor = AqwamTensorLibrary:createTensor(dimensionSizeArray, allValues)

	return AHAAutomaticDifferentiationTensor.new({tensor})

end

function AHAAutomaticDifferentiationTensor.createRandomNormalTensor(parameterDictionary)

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local mean = parameterDictionary.mean or parameterDictionary[2]

	local standardDeviation = parameterDictionary.standardDeviation or parameterDictionary[3]

	local tensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray, mean, standardDeviation)

	return AHAAutomaticDifferentiationTensor.new({tensor})

end

function AHAAutomaticDifferentiationTensor.createRandomUniformTensor(parameterDictionary)

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

function AHAAutomaticDifferentiationTensor:differentiate(derivativeTensor)

	if (not derivativeTensor) then

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self.tensor)

		derivativeTensor = AqwamTensorLibrary:createTensor(dimensionSizeArray, 1)

	end

	local PartialDerivativeFunction = self.PartialDerivativeFunction

	if (PartialDerivativeFunction) then PartialDerivativeFunction(derivativeTensor) end

	local totalDerivativeTensor = self.totalDerivativeTensor

	if (not totalDerivativeTensor) then

		totalDerivativeTensor = derivativeTensor

	else

		totalDerivativeTensor = AqwamTensorLibrary:add(totalDerivativeTensor, derivativeTensor)

	end

	self.totalDerivativeTensor = totalDerivativeTensor 

end

function AHAAutomaticDifferentiationTensor:copy()

	return deepCopyTable(self)

end

function AHAAutomaticDifferentiationTensor:getTensor(doNotDeepCopyTable)

	if (doNotDeepCopyTable) then

		return self.tensor

	else

		return deepCopyTable(self.tensor)

	end

end

function AHAAutomaticDifferentiationTensor:setTensor(tensor, doNotDeepCopyTable)

	if (doNotDeepCopyTable) then

		self.tensor = tensor

	else

		self.tensor = deepCopyTable(tensor)

	end

end

function AHAAutomaticDifferentiationTensor:getTotalDerivativeTensor(doNotDeepCopyTable)

	if (doNotDeepCopyTable) then 

		return self.totalDerivativeTensor

	else

		return deepCopyTable(self.totalDerivativeTensor)

	end

end

function AHAAutomaticDifferentiationTensor:setTotalDerivativeTensor(totalDerivativeTensor, doNotDeepCopyTable)

	if (doNotDeepCopyTable) then

		self.totalDerivativeTensor = totalDerivativeTensor

	else

		self.totalDerivativeTensor = deepCopyTable(totalDerivativeTensor)

	end

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:__tostring()

	return AqwamTensorLibrary:generateTensorString(self.tensor)

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

			return tensor

		end

	else

		return rawget(AHAAutomaticDifferentiationTensor, index)

	end

end

function AHAAutomaticDifferentiationTensor:__newindex(index, value)

	rawset(self, index, value)

end

function AHAAutomaticDifferentiationTensor:destroy(areDescendantsDestroyed)

	local inputTensorArray = self.inputTensorArray

	if (areDescendantsDestroyed) and (inputTensorArray) then

		for _, tensor in ipairs(inputTensorArray) do

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor) then tensor:destroy(true) end

		end

	end

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return AHAAutomaticDifferentiationTensor
