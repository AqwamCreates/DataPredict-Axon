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

local showFunctionErrorDueToNonObjectCondition = DisplayErrorFunctions.showFunctionErrorDueToNonObjectCondition

local Regularizer = {}

Regularizer.__index = Regularizer

local defaultLambda = 0.01

function Regularizer.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRegularizer = {}

	setmetatable(NewRegularizer, Regularizer)

	NewRegularizer.CalculateFunction = parameterDictionary.CalculateFunction or parameterDictionary[1]

	NewRegularizer.isAnObject = true

	return NewRegularizer

end

function Regularizer.ElasticNet(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local lambda = parameterDictionary.lambda or parameterDictionary[1] or defaultLambda
	
	local CalculateFunction = function(weightTensor)
		
		local signMatrix = AqwamTensorLibrary:applyFunction(math.sign, weightTensor)

		local regularizationMatrixPart1 = AqwamTensorLibrary:multiply(lambda, signMatrix)

		local regularizationMatrixPart2 = AqwamTensorLibrary:multiply(2, lambda, weightTensor)

		return AqwamTensorLibrary:add(regularizationMatrixPart1, regularizationMatrixPart2)
		
	end
	
	return Regularizer.new({CalculateFunction})
	
end

function Regularizer.Lasso(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local lambda = parameterDictionary.lambda or parameterDictionary[1] or defaultLambda

	local CalculateFunction = function(weightTensor)

		local signTensor = AqwamTensorLibrary:applyFunction(math.sign, weightTensor)

		return AqwamTensorLibrary:multiply(signTensor, lambda, weightTensor)

	end

	return Regularizer.new({CalculateFunction})

end

function Regularizer.Ridge(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local lambda = parameterDictionary.lambda or parameterDictionary[1] or defaultLambda

	local CalculateFunction = function(weightTensor)

		return AqwamTensorLibrary:multiply(2, lambda, weightTensor)

	end

	return Regularizer.new({CalculateFunction})

end

function Regularizer:calculate(parameterDictionary)
	
	showFunctionErrorDueToNonObjectCondition(self.isAnObject)
	
	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[1]

	if (self.CalculateFunction) then return self.CalculateFunction(weightTensor) end

end

function Regularizer:getLambda()
	
	showFunctionErrorDueToNonObjectCondition(self.isAnObject)

	return self.lambda

end

function Regularizer:setLambda(parameterDictionary)
	
	showFunctionErrorDueToNonObjectCondition(self.isAnObject)

	self.lambda = parameterDictionary.lambda or parameterDictionary[1]

end

return Regularizer
