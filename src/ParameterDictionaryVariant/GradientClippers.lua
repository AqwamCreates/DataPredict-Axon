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

local GradientClipper = {}

GradientClipper.__index = GradientClipper

local defaultDecayRate = 0.5

function GradientClipper.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewGradientClipper = {}

	setmetatable(NewGradientClipper, GradientClipper)
	
	NewGradientClipper.clipFunction = parameterDictionary.CalculateFunction or parameterDictionary[1]
	
	NewGradientClipper.Optimizer = parameterDictionary.Optimizer or parameterDictionary[2]
	
	NewGradientClipper.isAnObject = true
	
	return NewGradientClipper
	
end

function GradientClipper.ClipValue(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local minimumValue = parameterDictionary.minimumValue or parameterDictionary[1] or -1
	
	local maximumValue = parameterDictionary.maximumValue or parameterDictionary[2] or 1
	
	local Optimizer = parameterDictionary.Optimizer or parameterDictionary[3]
	
	local functionToApply = function(value) return math.clamp(value, minimumValue, maximumValue) end

	local clipFunction = function(firstDerivativeTensor) return AqwamTensorLibrary:applyFunction(functionToApply, firstDerivativeTensor) end

	return GradientClipper.new({clipFunction, Optimizer}) 

end

function GradientClipper.ClipNormalization(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local normalizationValue = parameterDictionary.normalizationValue or parameterDictionary[1] or 2

	local maximumNormalizationValue = parameterDictionary.maximumNormalizationValue or parameterDictionary[2] or normalizationValue

	local Optimizer = parameterDictionary.Optimizer or parameterDictionary[3]

	local clipFunction = function(firstDerivativeTensor) 

		local squaredFirstDerivativeTensor = AqwamTensorLibrary:power(firstDerivativeTensor, normalizationValue)

		local sumSquaredFirstDerivativeTensor = AqwamTensorLibrary:sum(squaredFirstDerivativeTensor)

		local currentNormalizationValue = math.pow(sumSquaredFirstDerivativeTensor, (1 / normalizationValue))

		if (currentNormalizationValue ~= 0) then

			firstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeTensor, (maximumNormalizationValue / currentNormalizationValue))

		end

		return firstDerivativeTensor
		
	end

	return GradientClipper.new({clipFunction, Optimizer}) 

end

function GradientClipper:calculate(parameterDictionary)
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	parameterDictionary = parameterDictionary or {}
	
	local learningRate = parameterDictionary.learningRate or parameterDictionary[1]

	local firstDerivativeTensor = parameterDictionary.firstDerivativeTensor or parameterDictionary[2]

	local clipFunction = self.clipFunction
	
	local Optimizer = self.Optimizer
	
	if (not clipFunction) then error("No calculate function.") end
	
	firstDerivativeTensor = clipFunction(firstDerivativeTensor)
	
	if (Optimizer) then
		
		firstDerivativeTensor = Optimizer:calculate(parameterDictionary)
		
	else
		
		firstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, firstDerivativeTensor)
		
	end
	
	return firstDerivativeTensor
	
end

function GradientClipper:reset()
	
	displayFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	local Optimizer = self.Optimizer
	
	if (Optimizer) then Optimizer:reset() end
	
end

return GradientClipper
