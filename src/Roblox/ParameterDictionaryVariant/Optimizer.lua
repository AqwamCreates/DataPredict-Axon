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

local Optimizer = {}

Optimizer.__index = Optimizer

local function showFunctionErrorDueToNonObjectCondition(showError)

	if (showError) then error("This function can only be called if it is an object.") end

end

local function calculateGaussianDensity(mean, standardDeviation)

	local exponentStep1 = math.pow(mean, 2)

	local exponentPart2 = math.pow(standardDeviation, 2)

	local exponentStep3 = exponentStep1 / exponentPart2

	local exponentStep4 = -0.5 * exponentStep3

	local exponentWithTerms = math.exp(exponentStep4)

	local divisor = standardDeviation * math.sqrt(2 * math.pi)

	local gaussianDensity = exponentWithTerms / divisor

	return gaussianDensity

end

function Optimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewWeightContainer = {}

	setmetatable(NewWeightContainer, Optimizer)
	
	NewWeightContainer.CalculateFunction = parameterDictionary.CalculateFunction or parameterDictionary[1]
	
	NewWeightContainer.optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[2]
	
	NewWeightContainer.LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[3]
	
	NewWeightContainer.isAnObject = true
	
	return NewWeightContainer
	
end

function Optimizer.AdaptiveGradient(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[1]
	
	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[2] or {}
	
	local CalculateFunction = function(learningRate, tensor)
		
		local previousSumOfGradientSquaredTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(tensor), 0)

		local gradientSquaredTensor = AqwamTensorLibrary:power(tensor, 2)

		local currentSumOfGradientSquaredTensor = AqwamTensorLibrary:add(previousSumOfGradientSquaredTensor, gradientSquaredTensor)

		local squareRootSumOfGradientSquaredTensor = AqwamTensorLibrary:power(currentSumOfGradientSquaredTensor, 0.5)

		local costFunctionDerivativeTensorPart1 = AqwamTensorLibrary:divide(tensor, squareRootSumOfGradientSquaredTensor)

		tensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeTensorPart1)
		
		optimizerInternalParameterArray[1] = currentSumOfGradientSquaredTensor
		
		return tensor
		
	end
	
	return Optimizer.new({CalculateFunction, optimizerInternalParameterArray, LearningRateValueScheduler})
	
end

function Optimizer.AdaptiveGradientDelta(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local decayRate = parameterDictionary.decayRate or parameterDictionary[1] or 0.9
	
	local epsilon = parameterDictionary.epsilon or parameterDictionary[2] or 1 * math.pow(10, -7)
	
	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[3]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[4] or {}

	local CalculateFunction = function(learningRate, tensor)

		local previousRunningGradientSquaredTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(tensor), 0)

		local gradientSquaredTensor = AqwamTensorLibrary:power(tensor, 2)

		local runningDeltaTensorPart1 = AqwamTensorLibrary:multiply(decayRate, previousRunningGradientSquaredTensor)

		local runningDeltaTensorPart2 = AqwamTensorLibrary:multiply((1 - decayRate), gradientSquaredTensor)

		local currentRunningGradientSquaredTensor =  AqwamTensorLibrary:add(runningDeltaTensorPart1, runningDeltaTensorPart2)

		local rootMeanSquareTensorPart1 = AqwamTensorLibrary:add(currentRunningGradientSquaredTensor, epsilon)

		local rootMeanSquareTensor = AqwamTensorLibrary:applyFunction(math.sqrt, rootMeanSquareTensorPart1)

		local tensorPart1 = AqwamTensorLibrary:divide(tensor, rootMeanSquareTensor)

		tensor = AqwamTensorLibrary:multiply(learningRate, tensorPart1)

		optimizerInternalParameterArray[1] = currentRunningGradientSquaredTensor

		return tensor

	end

	return Optimizer.new({CalculateFunction, optimizerInternalParameterArray, LearningRateValueScheduler})

end

function Optimizer.AdaptiveMomentEstimation(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local beta1 = parameterDictionary.beta1 or parameterDictionary[1] or 0.9
	
	local beta2 = parameterDictionary.beta2 or parameterDictionary[2] or 0.999

	local epsilon = parameterDictionary.epsilon or parameterDictionary[3] or 1 * math.pow(10, -7)

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[4]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[5] or {}

	local CalculateFunction = function(learningRate, tensor)

		local previousMomentumTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(tensor), 0)

		local previousVelocityTensor = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(tensor), 0)

		local momentumTensorPart1 = AqwamTensorLibrary:multiply(beta1, previousMomentumTensor)

		local momentumTensorPart2 = AqwamTensorLibrary:multiply((1 - beta1), tensor)

		local momentumTensor = AqwamTensorLibrary:add(momentumTensorPart1, momentumTensorPart2)

		local squaredCostFunctionDerivativeTensor = AqwamTensorLibrary:power(tensor, 2)

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(beta2, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply((1 - beta2), squaredCostFunctionDerivativeTensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		local meanMomentumTensor = AqwamTensorLibrary:divide(momentumTensor, (1 - beta1))

		local meanVelocityTensor = AqwamTensorLibrary:divide(velocityTensor, (1 - beta2))

		local squareRootedDivisor = AqwamTensorLibrary:power(meanVelocityTensor, 0.5)

		local finalDivisorTensor = AqwamTensorLibrary:add(squareRootedDivisor, epsilon)

		local tensorPart1 = AqwamTensorLibrary:divide(meanMomentumTensor, finalDivisorTensor)

		tensor = AqwamTensorLibrary:multiply(learningRate, tensorPart1)

		optimizerInternalParameterArray[1] = momentumTensor
		
		optimizerInternalParameterArray[2] = velocityTensor

		return tensor

	end

	return Optimizer.new({CalculateFunction, optimizerInternalParameterArray, LearningRateValueScheduler})

end

function Optimizer.AdaptiveMomentEstimationMaximum(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local beta1 = parameterDictionary.beta1 or parameterDictionary[1] or 0.9

	local beta2 = parameterDictionary.beta2 or parameterDictionary[2] or 0.999

	local epsilon = parameterDictionary.epsilon or parameterDictionary[3] or 1 * math.pow(10, -7)

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[4]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[5] or {}

	local CalculateFunction = function(learningRate, tensor)

		local momentTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(tensor), 0)

		local exponentWeightTensor = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(tensor), 0)

		local momentTensorPart1 = AqwamTensorLibrary:multiply(beta1, momentTensor)

		local momentTensorPart2 = AqwamTensorLibrary:multiply((1 - beta1), tensor)

		momentTensor = AqwamTensorLibrary:add(momentTensorPart1, momentTensorPart2)

		local exponentWeightTensorPart1 = AqwamTensorLibrary:multiply(beta2, exponentWeightTensor)

		local exponentWeightTensorPart2 = AqwamTensorLibrary:applyFunction(math.abs, tensor)

		exponentWeightTensor = AqwamTensorLibrary:applyFunction(math.max, exponentWeightTensorPart1, exponentWeightTensorPart2)

		local divisorTensorPart1 = 1 - math.pow(beta1, 2)

		local divisorTensorPart2 = AqwamTensorLibrary:add(exponentWeightTensor, epsilon)

		local divisorTensor = AqwamTensorLibrary:multiply(divisorTensorPart1, divisorTensorPart2)

		local tensorPart1 = AqwamTensorLibrary:divide(momentTensor, divisorTensor)

		tensor = AqwamTensorLibrary:multiply(learningRate, tensorPart1)

		optimizerInternalParameterArray[1] = momentTensor
		
		optimizerInternalParameterArray[2] = exponentWeightTensor

		return tensor

	end

	return Optimizer.new({CalculateFunction, optimizerInternalParameterArray, LearningRateValueScheduler})

end

function Optimizer.Gravity(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local initialStepSize = parameterDictionary.initialStepSize or parameterDictionary[1] or 0.01

	local movingAverage = parameterDictionary.movingAverage or parameterDictionary[2] or 0.9

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[3]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[4] or {}

	local CalculateFunction = function(learningRate, tensor)

		local previousVelocityTensor = optimizerInternalParameterArray[1]

		local currentTimeStep = optimizerInternalParameterArray[2] or 0

		currentTimeStep = currentTimeStep + 1

		if (not previousVelocityTensor) then

			local standardDeviation = initialStepSize / learningRate

			local gaussianDensity = calculateGaussianDensity(0, standardDeviation)

			previousVelocityTensor = AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(tensor), gaussianDensity)

		end

		local meanMovingAverage = ((movingAverage * currentTimeStep) + 1) / (currentTimeStep + 2)

		local absoluteMTensor = AqwamTensorLibrary:applyFunction(math.abs, tensor)

		local maxMTensor = AqwamTensorLibrary:findMaximumValue(absoluteMTensor)

		local mTensor = AqwamTensorLibrary:divide(1, maxMTensor)

		local weirdLTensorPart1 = AqwamTensorLibrary:divide(tensor, mTensor)

		local weirdLTensorPart2 = AqwamTensorLibrary:power(weirdLTensorPart1, 2)

		local weirdLTensorPart3 = AqwamTensorLibrary:add(1, weirdLTensorPart2)

		local weirdLTensor = AqwamTensorLibrary:divide(tensor, weirdLTensorPart3)

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(meanMovingAverage, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply((1 - meanMovingAverage), weirdLTensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		tensor = AqwamTensorLibrary:multiply(learningRate, velocityTensor) 

		optimizerInternalParameterArray[1] = velocityTensor
		
		optimizerInternalParameterArray[2] = currentTimeStep

		return tensor

	end

	return Optimizer.new({CalculateFunction, optimizerInternalParameterArray, LearningRateValueScheduler})

end

function Optimizer.Momentum(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local decayRate = parameterDictionary.decayRate or parameterDictionary[1] or 0.9

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[2]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[3] or {}

	local CalculateFunction = function(learningRate, tensor)

		local previousVelocityTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(tensor), 0)

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(decayRate, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply(learningRate, tensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		tensor = velocityTensor

		optimizerInternalParameterArray[1] = velocityTensor

		return tensor

	end

	return Optimizer.new({CalculateFunction, optimizerInternalParameterArray, LearningRateValueScheduler})

end

function Optimizer.NesterovAcceleratedAdaptiveMomentEstimation(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local beta1 = parameterDictionary.beta1 or parameterDictionary[1] or 0.9

	local beta2 = parameterDictionary.beta2 or parameterDictionary[2] or 0.999

	local epsilon = parameterDictionary.epsilon or parameterDictionary[3] or 1 * math.pow(10, -7)

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[4]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[5] or {}

	local CalculateFunction = function(learningRate, tensor)

		local previousMTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(tensor), 0)

		local previousNTensor = optimizerInternalParameterArray[2]  or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(tensor), 0)

		local meanCostFunctionDerivativeTensor = AqwamTensorLibrary:divide(tensor, (1 - beta1))

		local mTensorPart1 = AqwamTensorLibrary:multiply(beta1, previousMTensor)

		local mTensorPart2 = AqwamTensorLibrary:multiply((1 - beta1), tensor)

		local mTensor = AqwamTensorLibrary:add(mTensorPart1, mTensorPart2)

		local meanMTensor = AqwamTensorLibrary:divide(mTensor, (1 - beta1))

		local squaredCostFunctionDerivatives = AqwamTensorLibrary:power(tensor, 2)

		local nTensorPart1 = AqwamTensorLibrary:multiply(beta2, previousNTensor)

		local nTensorPart2 = AqwamTensorLibrary:multiply((1 - beta2), squaredCostFunctionDerivatives)

		local nTensor = AqwamTensorLibrary:add(nTensorPart1, nTensorPart2)

		local meanNTensor = AqwamTensorLibrary:divide(nTensor, (1 - beta2))

		local finalMTensorPart1 = AqwamTensorLibrary:multiply((1 - beta1), meanCostFunctionDerivativeTensor)

		local finalMTensorPart2 = AqwamTensorLibrary:multiply(beta1, meanMTensor)

		local finalMTensor = AqwamTensorLibrary:add(finalMTensorPart1, finalMTensorPart2)

		local squareRootedDivisor = AqwamTensorLibrary:power(meanNTensor, 0.5)

		local finalDivisor = AqwamTensorLibrary:add(squareRootedDivisor, epsilon)

		local tensorPart1 = AqwamTensorLibrary:divide(finalMTensor, finalDivisor)

		tensor = AqwamTensorLibrary:multiply(learningRate, tensorPart1)

		optimizerInternalParameterArray[1] = mTensor
		
		optimizerInternalParameterArray[2] = nTensor

		return tensor

	end

	return Optimizer.new({CalculateFunction, optimizerInternalParameterArray, LearningRateValueScheduler})

end

function Optimizer.RootMeanSquarePropagation(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local beta = parameterDictionary.beta or parameterDictionary[1] or 0.9

	local epsilon = parameterDictionary.epsilon or parameterDictionary[2] or 1 * math.pow(10, -7)

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[3]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[4] or {}

	local CalculateFunction = function(learningRate, tensor)

		local previousVelocity = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(tensor), 0)

		local squaredCostFunctionDerivativeTensor = AqwamTensorLibrary:power(tensor, 2)

		local vTensorPart1 = AqwamTensorLibrary:multiply(beta, previousVelocity)

		local vTensorPart2 = AqwamTensorLibrary:multiply((1 - beta), squaredCostFunctionDerivativeTensor)

		local velocityTensor = AqwamTensorLibrary:add(vTensorPart1, vTensorPart2)

		local velocityNonZeroDivisorTensor = AqwamTensorLibrary:add(velocityTensor, epsilon)

		local squaredRootVelocityTensor = AqwamTensorLibrary:power(velocityNonZeroDivisorTensor, 0.5)

		local tensorPart1 = AqwamTensorLibrary:divide(tensor, squaredRootVelocityTensor)

		tensor = AqwamTensorLibrary:multiply(learningRate, tensorPart1)

		optimizerInternalParameterArray[1] = velocityTensor

		return tensor

	end

	return Optimizer.new({CalculateFunction, optimizerInternalParameterArray, LearningRateValueScheduler})

end

function Optimizer.LearningRateStepDecay(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local timeStepToDecay = parameterDictionary.timeStepToDecay or parameterDictionary[1] or 100
	
	local decayRate = parameterDictionary.decayRate or parameterDictionary[2] or 0.5

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[3]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[4] or {}

	local CalculateFunction = function(learningRate, tensor)

		local currentLearningRate = optimizerInternalParameterArray[1] or learningRate

		local currentTimeStep = optimizerInternalParameterArray[2] or 0

		currentTimeStep = currentTimeStep + 1

		if ((currentTimeStep % timeStepToDecay) == 0) then

			currentLearningRate = currentLearningRate * decayRate

		end

		tensor = AqwamTensorLibrary:multiply(currentLearningRate, tensor)

		optimizerInternalParameterArray[1] = currentLearningRate
		
		optimizerInternalParameterArray[2] = currentTimeStep

		return tensor

	end

	return Optimizer.new({CalculateFunction, optimizerInternalParameterArray, LearningRateValueScheduler})
	
end

function Optimizer.LearningRateTimeDecay(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local decayRate = parameterDictionary.decayRate or parameterDictionary[1] or 0.5

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[2]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[3] or {}

	local CalculateFunction = function(learningRate, tensor)

		local currentLearningRate = optimizerInternalParameterArray[1] or learningRate

		local currentTimeStep = optimizerInternalParameterArray[2] or 0

		currentTimeStep = currentTimeStep + 1

		currentLearningRate = currentLearningRate / (decayRate * currentTimeStep)

		tensor = AqwamTensorLibrary:multiply(currentLearningRate, tensor)

		optimizerInternalParameterArray[1] = currentLearningRate
			
		optimizerInternalParameterArray[2] = currentTimeStep

		return tensor

	end

	return Optimizer.new({CalculateFunction, optimizerInternalParameterArray, LearningRateValueScheduler})

end

function Optimizer:calculate(parameterDictionary)
	
	showFunctionErrorDueToNonObjectCondition(not self.isAnObject)
	
	parameterDictionary = parameterDictionary or {}
	
	local learningRate = parameterDictionary.learningRate or parameterDictionary[1]
	
	local tensor = parameterDictionary.tensor or parameterDictionary[2]
	
	local CalculateFunction = self.CalculateFunction
	
	if (not CalculateFunction) then error("No calculate function.") end
	
	local LearningRateValueScheduler = self.LearningRateValueScheduler
	
	if (LearningRateValueScheduler) then learningRate = LearningRateValueScheduler:calculate{learningRate} end
	
	return CalculateFunction(learningRate, tensor)
	
end

return Optimizer
