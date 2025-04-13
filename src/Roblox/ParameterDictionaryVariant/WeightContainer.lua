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

function WeightContainer.new(parameterDictionary)
	
	local NewWeightContainer = {}

	setmetatable(NewWeightContainer, WeightContainer)
	
	NewWeightContainer.WeightTensorAndOptimizerArrayArray = parameterDictionary
	
	return NewWeightContainer
	
end

function WeightContainer:gradientDescent()
	
	for i, WeightTensorAndOptimizerArray in ipairs(self.WeightTensorAndOptimizerArrayArray) do
		
		local automaticDifferentiationTensor = WeightTensorAndOptimizerArray.automaticDifferentiationTensor or WeightTensorAndOptimizerArray[1]
		
		local learningRate =  WeightTensorAndOptimizerArray.learningRate or WeightTensorAndOptimizerArray[2] or defaultLearningRate
		
		local Optimizer = WeightTensorAndOptimizerArray.Optimizer or WeightTensorAndOptimizerArray[3]
		
		local tensorFirstDerivativeTensor = automaticDifferentiationTensor:getTotalFirstDerivativeTensor()
		
		local tensor = automaticDifferentiationTensor:getTensor()
		
		local optimizedFirstDerivativeTensor
		
		if (not tensorFirstDerivativeTensor) then error("Unable to find first derivative tensor for ADTensor " .. i .. ".") end
		
		if (Optimizer) then
			
			optimizedFirstDerivativeTensor = Optimizer:calculate(learningRate, tensorFirstDerivativeTensor)
			
		else
			
			optimizedFirstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, tensorFirstDerivativeTensor)
			
		end
		
		tensor = AqwamTensorLibrary:subtract(tensor, optimizedFirstDerivativeTensor)
		
		automaticDifferentiationTensor:setTotalFirstDerivativeTensor{nil, true}
		
		automaticDifferentiationTensor:setTensor{tensor, true}
		
	end
	
end

function WeightContainer:gradientAscent()
	
	for i, WeightTensorAndOptimizerArray in ipairs(self.WeightTensorAndOptimizerArrayArray) do

		local automaticDifferentiationTensor = WeightTensorAndOptimizerArray[1]

		local learningRate = WeightTensorAndOptimizerArray[2]

		local Optimizer = WeightTensorAndOptimizerArray[3]

		local tensorFirstDerivativeTensor = automaticDifferentiationTensor:getTotalFirstDerivativeTensor()

		local tensor = automaticDifferentiationTensor:getTensor()

		local optimizedFirstDerivativeTensor
		
		if (not tensorFirstDerivativeTensor) then error("Unable to find first derivative tensor for ADTensor " .. i .. ".") end

		if (Optimizer) then

			optimizedFirstDerivativeTensor = Optimizer:calculate(learningRate, tensorFirstDerivativeTensor)

		else

			optimizedFirstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, tensorFirstDerivativeTensor)

		end

		tensor = AqwamTensorLibrary:add(tensor, optimizedFirstDerivativeTensor)
		
		automaticDifferentiationTensor:setTotalFirstDerivativeTensor{nil, true}

		automaticDifferentiationTensor:setTensor{tensor, true}

	end
	
end

function WeightContainer:getWeightTensorArray(parameterDictionary)
	
	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[1]
	
	local weightTensorArray = {}
	
	for i, WeightTensorAndOptimizerArray in ipairs(self.WeightTensorAndOptimizerArrayArray) do

		local automaticDifferentiationTensor = WeightTensorAndOptimizerArray[1]
		
		weightTensorArray[i] = automaticDifferentiationTensor:getTensor(doNotDeepCopy)
		
	end
	
	return weightTensorArray
	
end

function WeightContainer:setWeightTensorArray(parameterDictionary)
	
	local weightTensorArray = parameterDictionary.weightTensorArray or parameterDictionary[1]
	
	local doNotDeepCopy = parameterDictionary.doNotDeepCopy or parameterDictionary[2]

	for i, WeightTensorAndOptimizerArray in ipairs(self.WeightTensorAndOptimizerArrayArray) do

		local automaticDifferentiationTensor = WeightTensorAndOptimizerArray[1]

		automaticDifferentiationTensor:getTensor(weightTensorArray[i], doNotDeepCopy)

	end

end

return WeightContainer