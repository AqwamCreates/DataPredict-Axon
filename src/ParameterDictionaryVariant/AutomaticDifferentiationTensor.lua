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

local AHAAutomaticDifferentiationTensor = {}

AHAAutomaticDifferentiationTensor.isFirstDerivativeTensorRequiredGlobally = true

AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally = true

AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false

local isWaitEnabledOnFirstDerivativeCalculation = false

local function register(wrapFunction, operationDictionary)

	local operatorFunction

	local derivativeFunction

	for name, functionDictionary in pairs(operationDictionary) do

		operatorFunction = functionDictionary.operatorFunction

		derivativeFunction = functionDictionary.derivativeFunction

		if (not operatorFunction) then error("No operator function for " .. name .. ".") end

		AHAAutomaticDifferentiationTensor[name] = wrapFunction(operatorFunction, derivativeFunction)

	end

end

local function checkIfCanDifferentiateTensor(tensor)
	
	if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return false end

	if (not tensor:getIsFirstDerivativeTensorRequired()) then return false end
	
	return true
	
end

local function deepCopyValue(original, copies)

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

				copy[deepCopyValue(originalKey, copies)] = deepCopyValue(originalValue, copies)

			end

			setmetatable(copy, deepCopyValue(getmetatable(original), copies))

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

--------------------------------------------------------------------------------------

local function createPartialDerivativeFunction(derivativeFunction, inputTensorArray, inputTensorValueArray, resultTensor)

	if (not AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) or (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false

		return

	end

	return function(firstDerivativeTensor)

		local partialFirstDerivativeTensor

		local dimensionSizeArray

		for tensorIndex, tensor in ipairs(inputTensorArray) do

			if checkIfCanDifferentiateTensor(tensor) then

				partialFirstDerivativeTensor = derivativeFunction(firstDerivativeTensor, inputTensorValueArray, resultTensor, tensorIndex)

				dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensorValueArray[tensorIndex])

				partialFirstDerivativeTensor = collapseTensor(partialFirstDerivativeTensor, dimensionSizeArray)

				tensor:differentiate{partialFirstDerivativeTensor}

			end

		end

	end

end

local function wrapUnaryOperation(operatorFunction, derivativeFunction)

	return function(parameterDictionary)

		local tensor = parameterDictionary.tensor or parameterDictionary[1]

		local inputTensorArray = {tensor}

		local tensorValue = AHAAutomaticDifferentiationTensor:fetchValue{tensor}

		local resultTensor = operatorFunction(tensorValue)

		local PartialFirstDerivativeFunction

		local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

		if (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

			PartialFirstDerivativeFunction = function(firstDerivativeTensor)

				local tensor = inputTensorArray[1]

				if (not checkIfCanDifferentiateTensor(tensor)) then return end

				local tensorValue = AHAAutomaticDifferentiationTensor:fetchValue{tensor}

				local partialDerivativeTensor = derivativeFunction(firstDerivativeTensor, tensorValue, resultTensor)

				tensor:differentiate{partialDerivativeTensor}

			end

		end

		if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

		return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

	end

end

local function wrapMetaMethodOperation(operatorFunction, derivativeFunction)

	return function(...)

		local inputTensorArray = {...}

		local inputTensorValueArray = {}

		for i, tensor in ipairs(inputTensorArray) do inputTensorValueArray[i] = AHAAutomaticDifferentiationTensor:fetchValue{tensor} end

		local resultTensor = operatorFunction(inputTensorValueArray)

		local PartialFirstDerivativeFunction = createPartialDerivativeFunction(derivativeFunction, inputTensorArray, inputTensorValueArray, resultTensor)

		return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})
		
	end

end

local function wrapOperation(operatorFunction, derivativeFunction)

	return function(parameterDictionary)

		local inputTensorArray = parameterDictionary

		local inputTensorValueArray = {}

		for i, tensor in ipairs(inputTensorArray) do inputTensorValueArray[i] = AHAAutomaticDifferentiationTensor:fetchValue{tensor} end

		local resultTensor = operatorFunction(inputTensorValueArray)
		
		local PartialFirstDerivativeFunction = createPartialDerivativeFunction(derivativeFunction, inputTensorArray, inputTensorValueArray, resultTensor)

		return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})
		
	end

end

local function wrapDimensionOperation(operationFunction, derivativeFunction)
	
	return function(self, parameterDictionary)
		
		displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
		
		parameterDictionary = parameterDictionary or {}
		
		local dimension = parameterDictionary.dimension or parameterDictionary[1]
		
		local inputTensorArray = {self}
		
		local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}
		
		local resultTensor = operationFunction(selfTensorValue, dimension)
		
		local PartialFirstDerivativeFunction

		if (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
			
			PartialFirstDerivativeFunction = function(firstDerivativeTensor) 
				
				firstDerivativeTensor = derivativeFunction(firstDerivativeTensor, selfTensorValue, dimension)

				inputTensorArray[1]:differentiate{firstDerivativeTensor}
				
			end
			
		end

		AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false

		return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})
		
	end
	
end

local function wrapDimensionArrayOperation(operationFunction, derivativeFunction)

	return function(self, parameterDictionary)

		displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

		parameterDictionary = parameterDictionary or {}

		local dimensionArray = parameterDictionary.dimensionArray or parameterDictionary[1]

		local inputTensorArray = {self}

		local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}

		local resultTensor = operationFunction(selfTensorValue, dimensionArray)

		local PartialFirstDerivativeFunction

		if (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

			PartialFirstDerivativeFunction = function(firstDerivativeTensor) 

				firstDerivativeTensor = derivativeFunction(firstDerivativeTensor, selfTensorValue, dimensionArray)

				inputTensorArray[1]:differentiate{firstDerivativeTensor}

			end

		end

		AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false

		return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

	end

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:enableRequireFirstDerivativeTensor()
	
	AHAAutomaticDifferentiationTensor.isFirstDerivativeTensorRequiredGlobally = true
	
end

function AHAAutomaticDifferentiationTensor:disableRequireFirstDerivativeTensor()

	AHAAutomaticDifferentiationTensor.isFirstDerivativeTensorRequiredGlobally = false

end

function AHAAutomaticDifferentiationTensor:enableFirstDerivativeCalculation()
	
	AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally = true
	
end

function AHAAutomaticDifferentiationTensor:disableFirstDerivativeCalculation()

	AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally = false

end

function AHAAutomaticDifferentiationTensor:disableFirstDerivativeCalculationForNextTensor()
	
	AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = true
	
end

function AHAAutomaticDifferentiationTensor:enableWaitOnFirstDerivativeCalculation()
	
	isWaitEnabledOnFirstDerivativeCalculation = true
	
end

function AHAAutomaticDifferentiationTensor:disableWaitOnFirstDerivativeCalculation()

	isWaitEnabledOnFirstDerivativeCalculation = false

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor.noFirstDerivativeCalculation(functionToWrap)
	
	local previousState = AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally -- For nested noFirstDerivativeCalculation functions.
	
	AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally = false
	
	functionToWrap()
	
	AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally = previousState
	
end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local isAutomaticDifferentiationTensor = pcall(function()

		tensor:isAutomaticDifferentiationTensor()

	end)

	return isAutomaticDifferentiationTensor

end

function AHAAutomaticDifferentiationTensor:getIsFirstDerivativeTensorRequired()
	
	return self.isFirstDerivativeTensorRequired
	
end

function AHAAutomaticDifferentiationTensor:fetchValue(parameterDictionary) -- DO NOT REMOVE THIS. I REPEAT. DO NOT REMOVE THIS AT ALL COSTS! THIS IS BECAUSE YOUR TENSOR LIBRARY CANNOT HANDLE AUTOMATIC DIFFERENTIATION TENSOR OBJECTS STORING A SCALAR VALUE!
	
	local automaticDifferentiationTensor = parameterDictionary.automaticDifferentiationTensor or parameterDictionary[1]

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
	
	self.isFirstDerivativeTensorRequired = parameterDictionary.isFirstDerivativeTensorRequired or parameterDictionary[4] or AHAAutomaticDifferentiationTensor.isFirstDerivativeTensorRequiredGlobally

	self.totalFirstDerivativeTensor = nil

	self.isAnObject = true

	return self

end

function AHAAutomaticDifferentiationTensor.coerce(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]
	
	if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then return tensor end

	return AHAAutomaticDifferentiationTensor.new({tensor, nil, {tensor}})

end

local unaryOperationDictionary = {
	
	absolute = {

		operatorFunction = function(tensor) return AqwamTensorLibrary:applyFunction(math.abs, tensor) end,
		derivativeFunction = function(firstDerivativeTensor, tensor) return AqwamTensorLibrary:applyFunction(function(firstDerivativeValue, value) return (((value >= 0) and firstDerivativeValue) or -firstDerivativeValue) end, firstDerivativeTensor, tensor) end,

	},
	
	rad = {
		
		operatorFunction = function(tensor) return AqwamTensorLibrary:applyFunction(math.rad, tensor) end,
		derivativeFunction = function(firstDerivativeTensor, tensor) return AqwamTensorLibrary:multiply((math.pi / 180), firstDerivativeTensor) end,
		
	},
	
	degree = {
		
		operatorFunction = function(tensor) return AqwamTensorLibrary:applyFunction(math.deg, tensor) end,
		derivativeFunction = function(firstDerivativeTensor, tensor) return AqwamTensorLibrary:multiply((180 / math.pi), firstDerivativeTensor) end,
		
	},
	
	sin = {
		
		operatorFunction = function(tensor) return AqwamTensorLibrary:applyFunction(math.sin, tensor) end,
		derivativeFunction = function(firstDerivativeTensor, tensor) return AqwamTensorLibrary:applyFunction(math.cos, firstDerivativeTensor, tensor) end
		
	},
	
	cos = {
		
		operatorFunction = function(tensor) return AqwamTensorLibrary:applyFunction(math.cos, tensor) end,
		derivativeFunction = function(firstDerivativeTensor, tensor) return AqwamTensorLibrary:applyFunction(function(firstDerivativeValue, value) return firstDerivativeValue * -math.sin(value) end, firstDerivativeTensor, tensor) end
		
	},
	
	tan = {

		operatorFunction = function(tensor) return AqwamTensorLibrary:applyFunction(math.tan, tensor) end,
		derivativeFunction = function(firstDerivativeTensor,tensor) return AqwamTensorLibrary:applyFunction(function(firstDerivativeValue, radianValue) return firstDerivativeValue * math.pow((1 / math.cos(radianValue)), 2) end, firstDerivativeTensor, tensor) end

	},
	
	exponent = {

		operatorFunction = function(tensor) return AqwamTensorLibrary:applyFunction(math.exp, tensor) end,
		derivativeFunction = function(firstDerivativeTensor, tensor) return AqwamTensorLibrary:multiply(firstDerivativeTensor, tensor) end

	},
	
	findMaximumValue = {

		operatorFunction = function(tensor) return AqwamTensorLibrary:findMaximumValue(tensor) end,
		derivativeFunction = function(firstDerivativeTensor, tensor, resultValue) return AqwamTensorLibrary:applyFunction(function(firstDerivativeValue, value) return ((value == resultValue) and value) or 0 end, firstDerivativeTensor, tensor) end

	},

	findMinimumValue = {

		operatorFunction = function(tensor) return AqwamTensorLibrary:findMinimumValue(tensor) end,
		derivativeFunction = function(firstDerivativeTensor, tensor, resultValue) return AqwamTensorLibrary:applyFunction(function(firstDerivativeValue, value) return ((value == resultValue) and value) or 0 end, firstDerivativeTensor, tensor) end

	}
	
}

local metaMethodOperationDictionary = {
	
	__eq = {

		operatorFunction = function(inputTensorArray) return AqwamTensorLibrary:applyFunction(function (value, otherValue) return ((value == otherValue) and 1) or 0 end, table.unpack(inputTensorArray)) end,

	},
	
	__ne = {

		operatorFunction = function(inputTensorArray) return AqwamTensorLibrary:applyFunction(function (value, otherValue) return ((value ~= otherValue) and 1) or 0 end, table.unpack(inputTensorArray)) end,

	},
	
	__lt = {

		operatorFunction = function(inputTensorArray) return AqwamTensorLibrary:applyFunction(function (value, otherValue) return ((value < otherValue) and 1) or 0 end, table.unpack(inputTensorArray)) end,

	},
	
	__le = {

		operatorFunction = function(inputTensorArray) return AqwamTensorLibrary:applyFunction(function (value, otherValue) return ((value <= otherValue) and 1) or 0 end, table.unpack(inputTensorArray)) end,

	},
	
	__gt = {

		operatorFunction = function(inputTensorArray) return AqwamTensorLibrary:applyFunction(function (value, otherValue) return ((value > otherValue) and 1) or 0 end, table.unpack(inputTensorArray)) end,

	},
	
	__ge = {
		
		operatorFunction = function(inputTensorArray) return AqwamTensorLibrary:applyFunction(function (value, otherValue) return ((value >= otherValue) and 1) or 0 end, table.unpack(inputTensorArray)) end,
		
	},
	
	__unm = {

		operatorFunction = function(inputTensorArray) return AqwamTensorLibrary:unaryMinus(inputTensorArray[1]) end,
		derivativeFunction = function(firstDerivativeTensor, inputTensorArray, resultTensor, tensorIndex) return AqwamTensorLibrary:unaryMinus(firstDerivativeTensor) end

	},
	
	__add = {

		operatorFunction = function(inputTensorArray) return AqwamTensorLibrary:add(table.unpack(inputTensorArray)) end,
		derivativeFunction = function(firstDerivativeTensor, inputTensorArray, resultTensor, tensorIndex) return firstDerivativeTensor end

	},
	
	__sub = {

		operatorFunction = function(inputTensorArray) return AqwamTensorLibrary:subtract(table.unpack(inputTensorArray)) end,
		derivativeFunction = function(firstDerivativeTensor, inputTensorArray, resultTensor, tensorIndex) return firstDerivativeTensor end

	},
	
	__mul = {

		operatorFunction = function(inputTensorArray) return AqwamTensorLibrary:multiply(table.unpack(inputTensorArray)) end,
		derivativeFunction = function(firstDerivativeTensor, inputTensorArray, resultTensor, tensorIndex) return AqwamTensorLibrary:multiply(inputTensorArray[((tensorIndex == 1) and 2) or 1], firstDerivativeTensor) end

	},
	
	__div = {

		operatorFunction = function(inputTensorArray) return AqwamTensorLibrary:divide(table.unpack(inputTensorArray)) end,
		derivativeFunction = function(firstDerivativeTensor, inputTensorArray, resultTensor, tensorIndex) return AqwamTensorLibrary:multiply(inputTensorArray[((tensorIndex == 1) and 2) or 1], firstDerivativeTensor) end

	},
	
	__pow = {

		operatorFunction = function(inputTensorArray) return AqwamTensorLibrary:power(table.unpack(inputTensorArray)) end,
		derivativeFunction = function(firstDerivativeTensor, inputTensorArray, resultTensor, tensorIndex)
			
			local baseTensor = inputTensorArray[1]
			
			local exponentTensor = inputTensorArray[2]
			
			local functionToApply
			
			if (tensorIndex == 1) then
				
				functionToApply = function(firstDerivativeValue, baseValue, exponentValue) return (firstDerivativeValue * exponentValue * math.pow(baseValue, (exponentValue - 1))) end
				
			else
				
				functionToApply = function(firstDerivativeValue, baseValue, exponentValue) return (firstDerivativeValue * math.pow(baseValue, exponentValue) * math.log(baseValue)) end
				
			end
			
			return AqwamTensorLibrary:applyFunction(functionToApply, firstDerivativeTensor, baseTensor, exponentTensor)
			
		end

	},
	
}

local operationDictionary = {
	
	stack = {

		operatorFunction = function(inputTensorArray)

			local resultTensor = {inputTensorArray[1]}

			for i = 2, #inputTensorArray, 1 do

				local previousTensor = inputTensorArray[i - 1]

				local currentTensor = inputTensorArray[i]

				local previousTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousTensor)

				local currentTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(currentTensor)

				local previousTensorNumberOfDimensions = #previousTensorDimensionSizeArray

				local currentTensorNumberOfDimensions = #currentTensorDimensionSizeArray

				if (previousTensorNumberOfDimensions ~= currentTensorNumberOfDimensions) then error("Tensor at " .. i .. " does not contain the same number of dimensions as the previous tensors.") end

				for j, size in ipairs(previousTensorDimensionSizeArray) do

					if (size ~= currentTensorDimensionSizeArray[j]) then error("Tensor at " .. i .. " does not contain the same dimension size at dimension " .. j .. " as the previous tensors.") end

				end

				resultTensor[i] = currentTensor

			end

			return resultTensor

		end,

		derivativeFunction = function(derivativeTensor, inputTensorArray, resultTensor, tensorIndex) return derivativeTensor[tensorIndex] end

	},

	maximum = {

		operatorFunction = function(inputTensorArray) 

			expandedPureTensorArray = {}

			expandedPureTensorArray[1] = inputTensorArray[1]

			for i = 2, #inputTensorArray, 1 do

				expandedPureTensorArray[i - 1], expandedPureTensorArray[i] = AqwamTensorLibrary:broadcast(inputTensorArray[i - 1], inputTensorArray[i])

			end
			
			return AqwamTensorLibrary:applyFunction(math.max, table.unpack(expandedPureTensorArray))

		end,

		derivativeFunction = function(firstDerivativeTensor, inputTensorArray, resultTensor, tensorIndex)

			local functionToApply = function(firstDerivativeValue, ...)

				local valueArray = {...} 

				local isMaximum = false

				local highestValue = -math.huge

				for j, value in ipairs(valueArray) do

					if (value >= highestValue) then

						isMaximum = (tensorIndex == j)

						highestValue = value

					end

				end

				return (isMaximum and firstDerivativeValue) or 0

			end

			return AqwamTensorLibrary:applyFunction(functionToApply, firstDerivativeTensor, table.unpack(expandedPureTensorArray))

		end

	},

	minimum = {

		operatorFunction = function(inputTensorArray) 

			expandedPureTensorArray = {}

			expandedPureTensorArray[1] = inputTensorArray[1]

			for i = 2, #inputTensorArray, 1 do

				expandedPureTensorArray[i - 1], expandedPureTensorArray[i] = AqwamTensorLibrary:broadcast(inputTensorArray[i - 1], inputTensorArray[i])

			end

			return AqwamTensorLibrary:applyFunction(math.max, table.unpack(expandedPureTensorArray))

		end,

		derivativeFunction = function(firstDerivativeTensor, inputTensorArray, resultTensor, tensorIndex)

			local functionToApply = function(firstDerivativeValue, ...)

				local valueArray = {...} 

				local isMinimum = false

				local lowestValue = math.huge

				for j, value in ipairs(valueArray) do

					if (value <= lowestValue) then

						isMinimum = (tensorIndex == j)

						lowestValue = value

					end

				end

				return (isMinimum and firstDerivativeValue) or 0

			end

			return AqwamTensorLibrary:applyFunction(functionToApply, firstDerivativeTensor, table.unpack(expandedPureTensorArray))

		end

	},
	
}

local dimensionOperationDictionary = {
	
	sum = {

		operatorFunction = function(tensor, dimension) return AqwamTensorLibrary:sum(tensor, dimension) end,
		
		derivativeFunction = function(firstDerivativeTensor, tensor, dimension)

			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

			if (dimension) then

				firstDerivativeTensor = AqwamTensorLibrary:expandDimensionSizes(firstDerivativeTensor, dimensionSizeArray)

			else

				firstDerivativeTensor = AqwamTensorLibrary:expandNumberOfDimensions(firstDerivativeTensor, dimensionSizeArray)

			end

			return firstDerivativeTensor 

		end

	},
	
	mean = {

		operatorFunction = function(tensor, dimension) return AqwamTensorLibrary:mean(tensor, dimension) end,
		derivativeFunction = function(firstDerivativeTensor, tensor, dimension)
			
			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)
			
			if (dimension) then

				totalNumberOfElements = dimensionSizeArray[dimension]

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, totalNumberOfElements)

			else

				totalNumberOfElements = 1

				for _, size in ipairs(dimensionSizeArray) do totalNumberOfElements = totalNumberOfElements * size end

				firstDerivativeTensor = AqwamTensorLibrary:createTensor(dimensionSizeArray, (firstDerivativeTensor / totalNumberOfElements))

			end
			
			
			return firstDerivativeTensor 
			
		end

	},
	
	standardDeviation = {

		operatorFunction = function(tensor, dimension) return AqwamTensorLibrary:standardDeviation(tensor, dimension) end,
		derivativeFunction = function(firstDerivativeTensor, tensor, dimension)
			
			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

			if (dimension) then

				totalNumberOfElements = dimensionSizeArray[dimension]

				firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, totalNumberOfElements)

			else

				totalNumberOfElements = 1

				for _, size in ipairs(dimensionSizeArray) do totalNumberOfElements = totalNumberOfElements * size end

				firstDerivativeTensor = AqwamTensorLibrary:createTensor(dimensionSizeArray, (firstDerivativeTensor / totalNumberOfElements))

			end

			return firstDerivativeTensor 

		end

	},
	
	zScoreNormalization = {

		operatorFunction = function(tensor, dimension) return AqwamTensorLibrary:zScoreNormalization(tensor, dimension) end,
		derivativeFunction = function(firstDerivativeTensor, tensor, dimension)

			local standardDeviationTensor = AqwamTensorLibrary:standardDeviation(tensor, dimension)

			local firstDerivativeTensor = AqwamTensorLibrary:divide(firstDerivativeTensor, standardDeviationTensor)

			return firstDerivativeTensor 

		end

	},
	
}

local dimensionArrayOperationDictionary = {
	
	transpose = {

		operatorFunction = function(tensor, dimensionArray) return AqwamTensorLibrary:transpose(tensor, dimensionArray) end,
		derivativeFunction = function(firstDerivativeTensor, tensor, dimensionArray) return AqwamTensorLibrary:transpose(firstDerivativeTensor, dimensionArray) end

	},
	
	flatten = {

		operatorFunction = function(tensor, dimensionArray) return AqwamTensorLibrary:flatten(tensor, dimensionArray) end,
		derivativeFunction = function(firstDerivativeTensor, tensor, dimensionArray) return AqwamTensorLibrary:reshape(firstDerivativeTensor, AqwamTensorLibrary:getDimensionSizeArray(tensor)) end

	},
	
	permute = {

		operatorFunction = function(tensor, dimensionArray) return AqwamTensorLibrary:permute(tensor, dimensionArray) end,
		derivativeFunction = function(firstDerivativeTensor, tensor, dimensionArray) return AqwamTensorLibrary:permute(firstDerivativeTensor, createOriginalDimensionArray(dimensionArray)) end

	},
	
}

register(wrapUnaryOperation, unaryOperationDictionary)
register(wrapMetaMethodOperation, metaMethodOperationDictionary)
register(wrapOperation, operationDictionary)
register(wrapDimensionOperation, dimensionOperationDictionary)
register(wrapDimensionArrayOperation, dimensionArrayOperationDictionary)

function AHAAutomaticDifferentiationTensor.logarithm(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local numberTensor = parameterDictionary.numberTensor or parameterDictionary[1]

	local baseTensor = parameterDictionary.baseTensor or parameterDictionary[2]
	
	local pureNumberTensor = AHAAutomaticDifferentiationTensor:fetchValue{numberTensor}
	
	local pureBaseTensor = AHAAutomaticDifferentiationTensor:fetchValue{baseTensor}

	local inputTensorArray = {numberTensor, baseTensor}

	local resultTensor = AqwamTensorLibrary:applyFunction(math.log, pureNumberTensor, pureBaseTensor)
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local numberTensor = inputTensorArray[1]

			local baseTensor = inputTensorArray[2]

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{numberTensor} then
				
				if (numberTensor:getIsFirstDerivativeTensorRequired()) then
					
					local pureNumberTensor = AHAAutomaticDifferentiationTensor:fetchValue{numberTensor}

					local numberTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureNumberTensor)

					local collapsedDerivativeTensor = collapseTensor(firstDerivativeTensor, numberTensorDimensionSizeArray)

					local partialDerivativeTensor

					if (baseTensor) then

						local partialDerivativeFunctionToApply = function (number, base) return (1 / (number * math.log(base))) end

						partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, pureNumberTensor, pureBaseTensor)

					else

						local partialDerivativeFunctionToApply = function (number) return (1 / number) end

						partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, pureNumberTensor)

					end

					numberTensor:differentiate{AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)}
					
				end
				
			end

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{baseTensor} then
				
				if (baseTensor:getIsFirstDerivativeTensorRequired()) then
					
					local pureBaseTensor = AHAAutomaticDifferentiationTensor:fetchValue{baseTensor}

					local baseTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureBaseTensor)

					local collapsedDerivativeTensor = collapseTensor(firstDerivativeTensor, baseTensorDimensionSizeArray)

					local partialDerivativeFunctionToApply = function (number, base) return -(math.log(number) / (base * math.pow(math.log(base), 2))) end

					local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, pureNumberTensor, baseTensorDimensionSizeArray)

					baseTensor:differentiate{AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)}
					
				end

			end

		end

	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.clamp(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local lowerBoundTensor = parameterDictionary.lowerBoundTensor or parameterDictionary[2]

	local upperBoundTensor = parameterDictionary.upperBoundTensor or parameterDictionary[3]
	
	local pureTensor = AHAAutomaticDifferentiationTensor:fetchValue{tensor}

	local pureLowerBoundTensor = AHAAutomaticDifferentiationTensor:fetchValue{lowerBoundTensor}
	
	local pureUpperBoundTensor = AHAAutomaticDifferentiationTensor:fetchValue{upperBoundTensor}

	local inputTensorArray = {tensor, lowerBoundTensor, upperBoundTensor}

	lowerBoundTensor = lowerBoundTensor or -math.huge

	upperBoundTensor = upperBoundTensor or math.huge

	local resultTensor = AqwamTensorLibrary:applyFunction(math.clamp, pureTensor, pureLowerBoundTensor, pureUpperBoundTensor)
	
	local PartialFirstDerivativeFunction
	
	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local tensor = inputTensorArray[1]

			if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end
			
			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end

			local pureTensor = AHAAutomaticDifferentiationTensor:fetchValue{tensor}

			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureTensor)

			local lowerBoundTensor = inputTensorArray[2]

			local upperBoundTensor = inputTensorArray[3]

			local functionToApply = function(value, derivative, lowerBoundValue, upperBoundValue) if ((value >= lowerBoundValue) and (value <= upperBoundValue)) then return value else return 0 end end

			local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, pureTensor, firstDerivativeTensor, pureLowerBoundTensor, pureUpperBoundTensor)

			local collapsedPartialDerivativeTensor = collapseTensor(partialDerivativeTensor, dimensionSizeArray)

			tensor:differentiate{collapsedPartialDerivativeTensor}

		end

	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end


function AHAAutomaticDifferentiationTensor:findMaximumValueDimensionIndexArray()

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local inputTensorArray = {self}
	
	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}
	
	return AHAAutomaticDifferentiationTensor:findMaximumValueDimensionIndexArray(selfTensorValue)

end

function AHAAutomaticDifferentiationTensor:findMinimumValueDimensionIndexArray()

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}

	return AHAAutomaticDifferentiationTensor:findMinimumValueDimensionIndexArray(selfTensorValue)

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:power(parameterDictionary)

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local otherTensor = parameterDictionary.otherTensor or parameterDictionary[1]

	local inputTensorArray = {self, otherTensor}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}

	local otherTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{otherTensor}

	local resultTensor = AqwamTensorLibrary:power(selfTensorValue, otherTensorValue)
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local selfTensor = inputTensorArray[1]

			local otherTensor = inputTensorArray[2]

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{selfTensor} then 
				
				if (selfTensor:getIsFirstDerivativeTensorRequired()) then
					
					local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{selfTensor}

					local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selfTensorValue)

					local chainRuleFirstDerivativeTensorPart1 = AqwamTensorLibrary:multiply(firstDerivativeTensor, otherTensor)

					local exponentMinusOneTensor = AqwamTensorLibrary:subtract(otherTensor, 1)

					local chainRuleFirstDerivativeTensorPart2 = AqwamTensorLibrary:power(selfTensor, exponentMinusOneTensor)

					local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(chainRuleFirstDerivativeTensorPart1, chainRuleFirstDerivativeTensorPart2)

					local collapsedChainRuleFirstDerivativeTensor = collapseTensor(chainRuleFirstDerivativeTensor, selfDimensionSizeArray)

					selfTensor:differentiate{collapsedChainRuleFirstDerivativeTensor}
					
				end

			end

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{otherTensor} then
				
				if (otherTensor:getIsFirstDerivativeTensorRequired()) then
					
					local otherTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{otherTensor}

					local otherTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(otherTensorValue)

					local partialFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(function(base, exponent) return (math.pow(base, exponent) * math.log(base)) end, self, otherTensorValue)

					local collapsedChainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(partialFirstDerivativeTensor, firstDerivativeTensor)

					local collapsedChainRuleDerivativeTensor = collapseTensor(collapsedChainRuleFirstDerivativeTensor, otherTensorDimensionSizeArray)

					otherTensor:differentiate{collapsedChainRuleFirstDerivativeTensor}
					
				end

			end

		end
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:dotProduct(parameterDictionary) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local otherTensor = parameterDictionary.otherTensor or parameterDictionary[1]

	local inputTensorArray = {self, otherTensor}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}

	local otherTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{otherTensor}

	local resultTensor = AqwamTensorLibrary:dotProduct(selfTensorValue, otherTensorValue)
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local selfTensor = inputTensorArray[1]

			local otherTensor = inputTensorArray[2]

			local selfTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selfTensorValue)

			local otherTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(otherTensorValue)

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{selfTensor} then
				
				if (selfTensor:getIsFirstDerivativeTensorRequired()) then
					
					local otherTensorNumberOfDimensions = #otherTensorDimensionSizeArray

					local transposedOther = AqwamTensorLibrary:transpose(otherTensorValue, {otherTensorNumberOfDimensions - 1, otherTensorNumberOfDimensions})

					local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(firstDerivativeTensor, transposedOther)

					local collapsedChainRuleFirstDerivativeTensor = collapseTensor(chainRuleFirstDerivativeTensor, selfTensorDimensionSizeArray)

					selfTensor:differentiate{collapsedChainRuleFirstDerivativeTensor} 
					
				end

			end

			if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{otherTensor} then
				
				if (otherTensor:getIsFirstDerivativeTensorRequired()) then
					
					local selfNumberOfDimensions = #selfTensorDimensionSizeArray

					local transposedSelf = AqwamTensorLibrary:transpose(selfTensorValue, {selfNumberOfDimensions - 1, selfNumberOfDimensions})

					local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(transposedSelf, firstDerivativeTensor)

					local collapsedChainRuleFirstDerivativeTensor = collapseTensor(chainRuleFirstDerivativeTensor, otherTensorDimensionSizeArray)

					otherTensor:differentiate{collapsedChainRuleFirstDerivativeTensor} 
					
				end
				
			end

		end
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:extract(parameterDictionary)

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local originDimensionIndexArray = parameterDictionary.originDimensionIndexArray or parameterDictionary[1]

	local targetDimensionIndexArray = parameterDictionary.targetDimensionIndexArray or parameterDictionary[2]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}

	local resultTensor = AqwamTensorLibrary:extract(selfTensorValue, originDimensionIndexArray, targetDimensionIndexArray)
	
	local PartialFirstDerivativeFunction
	
	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local tensor = inputTensorArray[1]

			if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end
			
			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end

			local originalTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selfTensorValue)

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
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor.concatenate(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensorArray = parameterDictionary.tensorArray or parameterDictionary

	local numberOfArguments = #tensorArray

	local dimensionIndex = tensorArray[numberOfArguments]

	if (type(dimensionIndex) ~= "number") then error("The final argument must be a number in order for it to be used as dimension index.") end

	table.remove(tensorArray, numberOfArguments)
	
	local pureTensorArray = {}

	for i = 1, #tensorArray, 1 do

		pureTensorArray[i] = AHAAutomaticDifferentiationTensor:fetchValue{tensorArray[i]}

	end

	local resultTensor

	for i, tensor in ipairs(pureTensorArray) do

		if (i > 1) then

			resultTensor = AqwamTensorLibrary:concatenate(resultTensor, tensor, dimensionIndex)

		else

			resultTensor = tensor

		end

	end
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local extractedDerivativeTensorArray = {}

			local firstDerivativeTensorDimensionArray = AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor)

			local originDimensionIndexArray = table.create(#firstDerivativeTensorDimensionArray, 1)

			local targetDimensionIndexArray = table.clone(firstDerivativeTensorDimensionArray)

			targetDimensionIndexArray[dimensionIndex] = 0

			for _, tensor in ipairs(pureTensorArray) do

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				targetDimensionIndexArray[dimensionIndex] = originDimensionIndexArray[dimensionIndex] + dimensionSizeArray[dimensionIndex] - 1

				local extractedDerivativeTensor = AqwamTensorLibrary:extract(firstDerivativeTensor, originDimensionIndexArray, targetDimensionIndexArray)

				originDimensionIndexArray[dimensionIndex] = originDimensionIndexArray[dimensionIndex] + dimensionSizeArray[dimensionIndex]

				table.insert(extractedDerivativeTensorArray, extractedDerivativeTensor)

			end

			for i, tensor in ipairs(tensorArray) do

				if AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then
					
					if (tensor:getIsFirstDerivativeTensorRequired()) then tensor:differentiate{extractedDerivativeTensorArray[i]}  end

				end

			end

		end

	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, tensorArray})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:reshape(parameterDictionary)

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}

	local resultTensor = AqwamTensorLibrary:reshape(selfTensorValue, dimensionSizeArray)
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)
			
			local tensor = inputTensorArray[1]

			if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end
			
			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end

			local pureTensor = AHAAutomaticDifferentiationTensor:fetchValue{tensor}

			local originalDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureTensor)

			firstDerivativeTensor = AqwamTensorLibrary:reshape(firstDerivativeTensor, originalDimensionSizeArray)

			tensor:differentiate{firstDerivativeTensor}

		end

	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:expandDimensionSizes(parameterDictionary)

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local targetDimensionSizeArray = parameterDictionary.targetDimensionSizeArray or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}

	local resultTensor = AqwamTensorLibrary:expandDimensionSizes(selfTensorValue, targetDimensionSizeArray)
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local tensor = inputTensorArray[1]

			if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end
			
			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end

			local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

			local chainRuleFirstDerivativeTensor = firstDerivativeTensor

			for dimension, dimensionSize in ipairs(tensorDimensionSizeArray) do

				if (dimensionSize == 1) and (targetDimensionSizeArray[dimension] > 1) then

					chainRuleFirstDerivativeTensor = AqwamTensorLibrary:sum(chainRuleFirstDerivativeTensor, dimension)

				end

			end

			tensor:differentiate{chainRuleFirstDerivativeTensor}

		end

	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	return AHAAutomaticDifferentiationTensor.new({resultTensor, PartialFirstDerivativeFunction, inputTensorArray})

end

function AHAAutomaticDifferentiationTensor:expandNumberOfDimensions(parameterDictionary)

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local dimensionSizeToAddArray = parameterDictionary.dimensionSizeToAddArray or parameterDictionary[1]

	local inputTensorArray = {self}

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}

	local resultTensor = AqwamTensorLibrary:expandNumberOfDimensions(selfTensorValue, dimensionSizeToAddArray)
	
	local PartialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then

		PartialFirstDerivativeFunction = function(firstDerivativeTensor)

			local tensor = inputTensorArray[1]

			if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor}) then return end
			
			if (not tensor:getIsFirstDerivativeTensorRequired()) then return end

			local numberOfDimensionsToSum = #dimensionSizeToAddArray

			local chainRuleFirstDerivativeTensor = firstDerivativeTensor

			for i = 1, numberOfDimensionsToSum, 1 do chainRuleFirstDerivativeTensor = AqwamTensorLibrary:sum(chainRuleFirstDerivativeTensor, 1)[1] end -- Remove the first dimension as it is redundant and does not carry any values. If it is not removed, this tensor might broadcast its dimension size elsewhere like during the gradient descent.

			tensor:differentiate{chainRuleFirstDerivativeTensor}

		end

	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AHAAutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

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

function AHAAutomaticDifferentiationTensor.createHeNormalTensor(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local numberOfInputNeurons = parameterDictionary.numberOfInputNeurons or parameterDictionary[2]

	local variancePart1 = 2 / numberOfInputNeurons

	local variancePart = math.sqrt(variancePart1)

	local randomNormalTensor = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)
	
	local tensor = AqwamTensorLibrary:multiply(variancePart, randomNormalTensor)

	return AHAAutomaticDifferentiationTensor.new({tensor})

end

function AHAAutomaticDifferentiationTensor.createHeUniformTensor(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local numberOfInputNeurons = parameterDictionary.numberOfInputNeurons or parameterDictionary[2]

	local variancePart1 = 6 / numberOfInputNeurons

	local variancePart = math.sqrt(variancePart1)

	local randomUniformTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

	local tensor = AqwamTensorLibrary:multiply(variancePart, randomUniformTensor) 

	return AHAAutomaticDifferentiationTensor.new({tensor})

end

function AHAAutomaticDifferentiationTensor.createXavierNormalTensor(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local numberOfInputNeurons = parameterDictionary.numberOfInputNeurons or parameterDictionary[2]
	
	local numberOfOutputNeurons = parameterDictionary.numberOfOutputNeurons or parameterDictionary[3]

	local variancePart1 = 2 / (numberOfInputNeurons + numberOfOutputNeurons)

	local variancePart = math.sqrt(variancePart1)

	local randomNormalTensor = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

	local tensor = AqwamTensorLibrary:multiply(variancePart, randomNormalTensor) 

	return AHAAutomaticDifferentiationTensor.new({tensor})

end

function AHAAutomaticDifferentiationTensor.createXavierUniformTensor(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local numberOfInputNeurons = parameterDictionary.numberOfInputNeurons or parameterDictionary[2]

	local numberOfOutputNeurons = parameterDictionary.numberOfOutputNeurons or parameterDictionary[3]

	local variancePart1 = 6 / (numberOfInputNeurons + numberOfOutputNeurons)

	local variancePart = math.sqrt(variancePart1)

	local randomUniformTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

	local tensor = AqwamTensorLibrary:multiply(variancePart, randomUniformTensor)

	return AHAAutomaticDifferentiationTensor.new({tensor})

end

function AHAAutomaticDifferentiationTensor.createLeCunNormalTensor(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local numberOfInputNeurons = parameterDictionary.numberOfInputNeurons or parameterDictionary[2]

	local variancePart1 = 1 / numberOfInputNeurons

	local variancePart = math.sqrt(variancePart1)

	local randomNormalTensor = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

	local tensor = AqwamTensorLibrary:multiply(variancePart, randomNormalTensor)
	
	return AHAAutomaticDifferentiationTensor.new({tensor})

end

function AHAAutomaticDifferentiationTensor.createLeCunUniformTensor(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray or parameterDictionary[1]

	local numberOfInputNeurons = parameterDictionary.numberOfInputNeurons or parameterDictionary[2]

	local variancePart1 = 3 / numberOfInputNeurons

	local variancePart = math.sqrt(variancePart1)

	local randomUniformTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

	local tensor = AqwamTensorLibrary:multiply(variancePart, randomUniformTensor) 

	return AHAAutomaticDifferentiationTensor.new({tensor})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:sample(parameterDictionary)

	parameterDictionary = parameterDictionary or {}
	
	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}

	local dimension = parameterDictionary.dimension or parameterDictionary[1]
	
	local inputTensorArray = {self}
	
	local indexTensor = AqwamTensorLibrary:sample(selfTensorValue, dimension)

	return AHAAutomaticDifferentiationTensor.new({indexTensor, nil, inputTensorArray})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:isAutomaticDifferentiationTensor()

	return true

end

function AHAAutomaticDifferentiationTensor:isScalar()

	return (type(AHAAutomaticDifferentiationTensor:fetchValue{self}) == "number")

end

function AHAAutomaticDifferentiationTensor:differentiate(parameterDictionary)

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local firstDerivativeTensor = parameterDictionary.firstDerivativeTensor or parameterDictionary[1]

	local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}

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
	
	if (isWaitEnabledOnFirstDerivativeCalculation) then task.wait() end

	if (PartialFirstDerivativeFunction) then PartialFirstDerivativeFunction(firstDerivativeTensor) end

	local totalFirstDerivativeTensor = self.totalFirstDerivativeTensor

	if (not totalFirstDerivativeTensor) then

		totalFirstDerivativeTensor = firstDerivativeTensor

	else

		totalFirstDerivativeTensor = AqwamTensorLibrary:add(totalFirstDerivativeTensor, firstDerivativeTensor)

	end

	self.totalFirstDerivativeTensor = totalFirstDerivativeTensor

end

function AHAAutomaticDifferentiationTensor:detach()
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	return AHAAutomaticDifferentiationTensor.new{self.tensor, nil, self.inputTensorArray}
	
end

function AHAAutomaticDifferentiationTensor:getDimensionSizeArray()
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	return AqwamTensorLibrary:getDimensionSizeArray(self.tensor)
	
end

function AHAAutomaticDifferentiationTensor:copy()

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	return deepCopyValue(self)

end

function AHAAutomaticDifferentiationTensor:getTensor(parameterDictionary)

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[1]

	if (doNotDeepCopy) then

		return self.tensor

	else

		return deepCopyValue(self.tensor)

	end

end

function AHAAutomaticDifferentiationTensor:setTensor(parameterDictionary)

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[2]

	if (doNotDeepCopy) then

		self.tensor = tensor

	else

		self.tensor = deepCopyValue(tensor)

	end

end

function AHAAutomaticDifferentiationTensor:getTotalFirstDerivativeTensor(parameterDictionary)

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[1]

	if (doNotDeepCopy) then 

		return self.totalFirstDerivativeTensor

	else

		return deepCopyValue(self.totalFirstDerivativeTensor)

	end

end

function AHAAutomaticDifferentiationTensor:setTotalFirstDerivativeTensor(parameterDictionary)

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

	parameterDictionary = parameterDictionary or {}

	local totalFirstDerivativeTensor = parameterDictionary.totalFirstDerivativeTensor or parameterDictionary[1]

	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[2]

	if (doNotDeepCopy) then

		self.totalFirstDerivativeTensor = totalFirstDerivativeTensor

	else

		self.totalFirstDerivativeTensor = deepCopyValue(totalFirstDerivativeTensor)

	end

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:__tostring()

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

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

		local selfTensorValue = AHAAutomaticDifferentiationTensor:fetchValue{self}

		if (type(selfTensorValue) == "table") then
			
			local selfSubTensorValue = selfTensorValue[index]

			local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

				if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{self}) then return end
				
				local targetDimensionSize = AqwamTensorLibrary:getDimensionSizeArray(selfTensorValue)[1]
				
				local headPaddingDimensionSize = index - 1

				local tailPaddingDimensionSize = targetDimensionSize - index
				
				firstDerivativeTensor = {firstDerivativeTensor}

				local firstDerivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor)

				if (headPaddingDimensionSize >= 1) then

					local tensorHeadPaddingDimensionSizeArray = table.clone(firstDerivativeTensorDimensionSizeArray)

					tensorHeadPaddingDimensionSizeArray[1] = headPaddingDimensionSize

					local headPaddingTensor = AqwamTensorLibrary:createTensor(tensorHeadPaddingDimensionSizeArray)

					firstDerivativeTensor = AqwamTensorLibrary:concatenate(headPaddingTensor, firstDerivativeTensor, 1)

				end

				if (tailPaddingDimensionSize >= 1) then

					local tensorTailPaddingDimensionSizeArray = table.clone(firstDerivativeTensorDimensionSizeArray)

					tensorTailPaddingDimensionSizeArray[1] = tailPaddingDimensionSize

					local tailPaddingTensor = AqwamTensorLibrary:createTensor(tensorTailPaddingDimensionSizeArray)

					firstDerivativeTensor = AqwamTensorLibrary:concatenate(firstDerivativeTensor, tailPaddingTensor, 1)

				end

				self:differentiate{firstDerivativeTensor}

			end

			return AHAAutomaticDifferentiationTensor.new({selfSubTensorValue, PartialFirstDerivativeFunction, {self}})

		else
			
			error("Unable to index ADTensor number value with a number.")

		end

	else

		return rawget(AHAAutomaticDifferentiationTensor, index)

	end

end

function AHAAutomaticDifferentiationTensor:__newindex(index, value)

	rawset(self, index, value)

end

function AHAAutomaticDifferentiationTensor:getString()
	
	local tensor = self.tensor
	
	local pureTensor = AHAAutomaticDifferentiationTensor:fetchValue{tensor, true}
	
	if (type(pureTensor) == "number") then return tostring(pureTensor) end
	
	return AqwamTensorLibrary:generateTensorWithCommaString(pureTensor)
	
end

function AHAAutomaticDifferentiationTensor:destroy(parameterDictionary)

	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)

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
