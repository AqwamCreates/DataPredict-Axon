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
	
	NewWeightContainer.TensorAndOptimizerArrayArray = parameterDictionary
	
	return NewWeightContainer
	
end

function WeightContainer:gradientDescent()
	
	for i, TensorAndOptimizerArray in ipairs(self.TensorAndOptimizerArrayArray) do
		
		local automaticDifferentiationTensor = TensorAndOptimizerArray.automaticDifferentiationTensor or TensorAndOptimizerArray[1]
		
		local learningRate =  TensorAndOptimizerArray.learningRate or TensorAndOptimizerArray[2] or defaultLearningRate
		
		local Optimizer = TensorAndOptimizerArray.Optimizer or TensorAndOptimizerArray[3]
		
		local firstDerivativeTensor = automaticDifferentiationTensor:getTotalFirstDerivativeTensor()
		
		local tensor = automaticDifferentiationTensor:getTensor()
		
		local optimizedFirstDerivativeTensor
		
		if (not firstDerivativeTensor) then error("Unable to find first derivative tensor for ADTensor " .. i .. ".") end
		
		if (Optimizer) then
			
			optimizedFirstDerivativeTensor = Optimizer:calculate{learningRate, firstDerivativeTensor}
			
		else
			
			optimizedFirstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, firstDerivativeTensor)
			
		end
		
		tensor = AqwamTensorLibrary:subtract(tensor, optimizedFirstDerivativeTensor)
		
		automaticDifferentiationTensor:setTotalFirstDerivativeTensor{nil, true}
		
		automaticDifferentiationTensor:setTensor{tensor, true}
		
	end
	
end

function WeightContainer:gradientAscent()
	
	for i, TensorAndOptimizerArray in ipairs(self.TensorAndOptimizerArrayArray) do

		local automaticDifferentiationTensor = TensorAndOptimizerArray[1]

		local learningRate = TensorAndOptimizerArray[2]

		local Optimizer = TensorAndOptimizerArray[3]

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