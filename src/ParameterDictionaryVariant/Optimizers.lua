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

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultWeightDecayRate = 0

local defaultEpsilon = 1e-16

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
	
	local NewOptimizer = {}

	setmetatable(NewOptimizer, Optimizer)
	
	NewOptimizer.CalculateFunction = parameterDictionary.CalculateFunction or parameterDictionary[1]
	
	NewOptimizer.LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[2]
	
	NewOptimizer.optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[3]
	
	NewOptimizer.isAnObject = true
	
	return NewOptimizer
	
end

function Optimizer.AdaptiveDelta(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local decayRate = parameterDictionary.decayRate or parameterDictionary[1] or 0.9
	
	local weightDecayRate = parameterDictionary.decayRate or parameterDictionary[2] or defaultWeightDecayRate
	
	local epsilon = parameterDictionary.epsilon or parameterDictionary[3] or defaultEpsilon
	
	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[4]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[5] or {}

	local CalculateFunction = function(learningRate, firstDerivativeTensor, tensor)

		local previousRunningGradientSquaredTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0)

		local gradientTensor = firstDerivativeTensor

		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, tensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end

		local gradientSquaredTensor = AqwamTensorLibrary:power(gradientTensor, 2)

		local runningDeltaTensorPart1 = AqwamTensorLibrary:multiply(decayRate, previousRunningGradientSquaredTensor)

		local runningDeltaTensorPart2 = AqwamTensorLibrary:multiply((1 - decayRate), gradientSquaredTensor)

		local currentRunningGradientSquaredTensor = AqwamTensorLibrary:add(runningDeltaTensorPart1, runningDeltaTensorPart2)

		local rootMeanSquareTensorPart1 = AqwamTensorLibrary:add(currentRunningGradientSquaredTensor, epsilon)

		local rootMeanSquareTensor = AqwamTensorLibrary:applyFunction(math.sqrt, rootMeanSquareTensorPart1)

		local firstDerivativeTensorPart1 = AqwamTensorLibrary:divide(gradientTensor, rootMeanSquareTensor)

		firstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, firstDerivativeTensorPart1)

		optimizerInternalParameterArray[1] = currentRunningGradientSquaredTensor

		return firstDerivativeTensor

	end

	return Optimizer.new({CalculateFunction, LearningRateValueScheduler, optimizerInternalParameterArray})

end

function Optimizer.AdaptiveFactor(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local beta2DecayRate = parameterDictionary.beta2DecayRate or parameterDictionary[1] or -0.8

	local weightDecayRate = parameterDictionary.weightDecayRate or parameterDictionary[2] or defaultWeightDecayRate

	local clipValue = parameterDictionary.clipValue or parameterDictionary[3] or 1

	local epsilon1 = parameterDictionary.epsilon1 or parameterDictionary[4] or epsilon

	local epsilon2 = parameterDictionary.epsilon2 or parameterDictionary[5] or epsilon

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[6]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[7] or {}

	local CalculateFunction = function(learningRate, firstDerivativeTensor, tensor)

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor)

		local secondMomentRowFactorTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)

		local secondMomentColumnFactorTensor = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)

		local timeValue = (optimizerInternalParameterArray[3] or 0) 1

		local beta2 = 1 - math.pow(timeValue, beta2DecayRate)

		local oneMinusBeta2 = 1 - beta2

		local gradientTensor = costFunctionDerivativeTensor

		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, tensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end

		local squaredGradientTensor = AqwamTensorLibrary:power(gradientTensor, 2)

		local oneRowTensor = AqwamTensorLibrary:createTensor({dimensionSizeArray[1], 1}, 1)

		local oneColumnTensor = AqwamTensorLibrary:createTensor({dimensionSizeArray[2], 1}, 1)

		local transposedOneRowTensor = AqwamTensorLibrary:transpose(oneRowTensor)

		local transposedOneColumnTensor = AqwamTensorLibrary:transpose(oneColumnTensor)

		local dotProductOnTensor = AqwamTensorLibrary:dotProduct(oneRowTensor, transposedOneColumnTensor)

		local epsilonMultiplyDotProductOnTensor = AqwamTensorLibrary:multiply(epsilon1, dotProductOnTensor)

		local squaredGradientAddEpsilonMultiplyDotProductOnTensor = AqwamTensorLibrary:add(squaredGradientTensor, epsilonMultiplyDotProductOnTensor)

		local secondMomentRowFactorTensorPart1 = AqwamTensorLibrary:multiply(beta2, secondMomentRowFactorTensor)

		local secondMomentRowFactorTensorPart2 = AqwamTensorLibrary:multiply(oneMinusBeta2, squaredGradientAddEpsilonMultiplyDotProductOnTensor)

		local secondMomentRowFactorTensorPart3 = AqwamTensorLibrary:dotProduct(secondMomentRowFactorTensorPart2, oneColumnTensor)

		secondMomentRowFactorTensor = AqwamTensorLibrary:add(secondMomentRowFactorTensorPart1, secondMomentRowFactorTensorPart3)

		local secondMomentColumnFactorTensorPart1 = AqwamTensorLibrary:multiply(beta2, secondMomentColumnFactorTensor)

		local secondMomentColumnFactorTensorPart2 = AqwamTensorLibrary:dotProduct(transposedOneRowTensor, squaredGradientAddEpsilonMultiplyDotProductOnTensor)

		local secondMomentColumnFactorTensorPart3 = AqwamTensorLibrary:multiply(oneMinusBeta2, secondMomentRowFactorTensorPart2)

		secondMomentColumnFactorTensor = AqwamTensorLibrary:add(secondMomentColumnFactorTensorPart1, secondMomentColumnFactorTensorPart3)

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(secondMomentRowFactorTensor, secondMomentColumnFactorTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:dotProduct(transposedOneRowTensor, secondMomentRowFactorTensor)

		local velocityTensor = AqwamTensorLibrary:divide(velocityTensorPart1, velocityTensorPart2)

		local uTensor = AqwamTensorLibrary:divide(gradientTensor, AqwamTensorLibrary:applyFunction(math.sqrt, velocityTensor))

		local squareRootVelocityTensor = AqwamTensorLibrary:applyFunction(math.sqrt, velocityTensor)

		local dividedRootMeanSquaredXTensor = AqwamTensorLibrary:divide(uTensor, weightTensor)

		local momentum = math.min(learningRate, (1 / math.sqrt(timeValue)))

		local alpha = AqwamTensorLibrary:applyFunction(math.max, {{epsilon2}}, dividedRootMeanSquaredXTensor)

		alpha = AqwamTensorLibrary:multiply(alpha, momentum)

		local rootMeanSquaredUTensorPart1 = AqwamTensorLibrary:divide(gradientTensor, squareRootVelocityTensor)

		local rootMeanSquaredUTensor = AqwamTensorLibrary:unaryMinus(rootMeanSquaredUTensorPart1)

		local dividedRootMeanSquaredUTensor = AqwamTensorLibrary:divide(rootMeanSquaredUTensor, clipValue)

		local finalUTensor = AqwamTensorLibrary:divide(uTensor, AqwamTensorLibrary:applyFunction(math.max, dividedRootMeanSquaredUTensor, 1))

		firstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, finalUTensor)

		optimizerInternalParameterArray[1] = secondMomentRowFactorTensor
		
		optimizerInternalParameterArray[2] = secondMomentColumnFactorTensor
		
		optimizerInternalParameterArray[3] = timeValue
		
		return firstDerivativeTensor

	end

	return Optimizer.new({CalculateFunction, LearningRateValueScheduler, optimizerInternalParameterArray})

end

function Optimizer.AdaptiveGradient(parameterDictionary)

	parameterDictionary = parameterDictionary or {}
	
	local weightDecayRate = parameterDictionary.weightDecayRate or parameterDictionary[1] or defaultWeightDecayRate
	
	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[2]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[3] or {}

	local CalculateFunction = function(learningRate, firstDerivativeTensor, tensor)

		local previousSumOfGradientSquaredTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0)

		local gradientTensor = firstDerivativeTensor

		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, tensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end

		local gradientSquaredTensor = AqwamTensorLibrary:power(gradientTensor, 2)

		local currentSumOfGradientSquaredTensor = AqwamTensorLibrary:add(previousSumOfGradientSquaredTensor, gradientSquaredTensor)

		local squareRootSumOfGradientSquaredTensor = AqwamTensorLibrary:applyFunction(math.sqrt, currentSumOfGradientSquaredTensor)

		local firstDerivativeTensorPart1 = AqwamTensorLibrary:divide(gradientTensor, squareRootSumOfGradientSquaredTensor)

		firstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, firstDerivativeTensorPart1)

		optimizerInternalParameterArray[1] = currentSumOfGradientSquaredTensor

		return firstDerivativeTensor

	end

	return Optimizer.new({CalculateFunction, LearningRateValueScheduler, optimizerInternalParameterArray})

end

function Optimizer.AdaptiveMomentEstimation(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local beta1 = parameterDictionary.beta1 or parameterDictionary[1] or defaultBeta1
	
	local beta2 = parameterDictionary.beta2 or parameterDictionary[2] or defaultBeta2
	
	local weightDecayRate = parameterDictionary.weightDecayRate or parameterDictionary[3] or defaultWeightDecayRate

	local epsilon = parameterDictionary.epsilon or parameterDictionary[4] or defaultEpsilon

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[5]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[6] or {}

	local CalculateFunction = function(learningRate, firstDerivativeTensor, tensor)

		local previousMomentumTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0)

		local previousVelocityTensor = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0)

		local timeValue = (optimizerInternalParameterArray[3] or 0) 1

		local gradientTensor = firstDerivativeTensor

		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, tensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end

		local momentumTensorPart1 = AqwamTensorLibrary:multiply(beta1, previousMomentumTensor)

		local momentumTensorPart2 = AqwamTensorLibrary:multiply((1 - beta1), gradientTensor)

		local momentumTensor = AqwamTensorLibrary:add(momentumTensorPart1, momentumTensorPart2)

		local squaredGradientDerivativeTensor = AqwamTensorLibrary:power(gradientTensor, 2)

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(beta2, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply((1 - beta2), squaredGradientDerivativeTensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		local meanMomentumTensor = AqwamTensorLibrary:divide(momentumTensor, (1 - math.pow(beta1, timeValue)))

		local meanVelocityTensor = AqwamTensorLibrary:divide(velocityTensor, (1 - math.pow(beta2, timeValue)))

		local squareRootedDivisor = AqwamTensorLibrary:applyFunction(math.sqrt, meanVelocityTensor)

		local finalDivisorTensor = AqwamTensorLibrary:add(squareRootedDivisor, epsilon)

		local firstDerivativeTensorPart1 = AqwamTensorLibrary:divide(meanMomentumTensor, finalDivisorTensor)

		firstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, firstDerivativeTensorPart1)

		timeValue = timeValue + 1

		optimizerInternalParameterArray[1] = momentumTensor
		
		optimizerInternalParameterArray[2] = velocityTensor
		
		optimizerInternalParameterArray[3] = timeValue

		return firstDerivativeTensor

	end

	return Optimizer.new({CalculateFunction, LearningRateValueScheduler, optimizerInternalParameterArray})

end

function Optimizer.AdaptiveMomentEstimationMaximum(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local beta1 = parameterDictionary.beta1 or parameterDictionary[1] or defaultBeta1

	local beta2 = parameterDictionary.beta2 or parameterDictionary[2] or defaultBeta2

	local weightDecayRate = parameterDictionary.weightDecayRate or parameterDictionary[3] or defaultWeightDecayRate

	local epsilon = parameterDictionary.epsilon or parameterDictionary[4] or defaultEpsilon

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[5]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[6] or {}

	local CalculateFunction = function(learningRate, firstDerivativeTensor, tensor)

		local momentTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0)

		local exponentWeightTensor = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0)

		local timeValue = (optimizerInternalParameterArray[3] or 0) + 1

		local gradientTensor = firstDerivativeTensor

		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, tensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end

		local momentTensorPart1 = AqwamTensorLibrary:multiply(beta1, momentTensor)

		local momentTensorPart2 = AqwamTensorLibrary:multiply((1 - beta1), gradientTensor)

		momentTensor = AqwamTensorLibrary:add(momentTensorPart1, momentTensorPart2)

		local exponentWeightTensorPart1 = AqwamTensorLibrary:multiply(beta2, exponentWeightTensor)

		local exponentWeightTensorPart2 = AqwamTensorLibrary:applyFunction(math.abs, gradientTensor)

		exponentWeightTensor = AqwamTensorLibrary:applyFunction(math.max, exponentWeightTensorPart1, exponentWeightTensorPart2)

		local divisorTensorPart1 = 1 - math.pow(beta1, timeValue)

		local divisorTensorPart2 = AqwamTensorLibrary:add(exponentWeightTensor, epsilon)

		local divisorTensor = AqwamTensorLibrary:multiply(divisorTensorPart1, divisorTensorPart2)

		local firstDerivativeTensorPart1 = AqwamTensorLibrary:divide(momentTensor, divisorTensor)

		firstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, firstDerivativeTensorPart1)

		optimizerInternalParameterArray[1] = momentTensor
		
		optimizerInternalParameterArray[2] = exponentWeightTensor
		
		optimizerInternalParameterArray[3] = timeValue

		return firstDerivativeTensor

	end

	return Optimizer.new({CalculateFunction, LearningRateValueScheduler, optimizerInternalParameterArray})

end

function Optimizer.Gravity(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local initialStepSize = parameterDictionary.initialStepSize or parameterDictionary[1] or 0.01

	local movingAverage = parameterDictionary.movingAverage or parameterDictionary[2] or 0.9

	local weightDecayRate = parameterDictionary.weightDecayRate or parameterDictionary[3] or defaultWeightDecayRate

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[4]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[5] or {}

	local CalculateFunction = function(learningRate, firstDerivativeTensor, tensor)

		local previousVelocityTensor = optimizerInternalParameterArray[1]

		local timeValue = (optimizerInternalParameterArray[2] or 0) + 1

		if (not previousVelocityTensor) then

			local standardDeviation = initialStepSize / learningRate

			local gaussianDensity = calculateGaussianDensity(0, standardDeviation)

			previousVelocityTensor = AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), gaussianDensity)

		end

		local gradientTensor = firstDerivativeTensor

		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, tensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end

		local meanMovingAverage = ((movingAverage * timeValue) + 1) / (timeValue + 2)

		local absoluteGradientTensor = AqwamTensorLibrary:applyFunction(math.abs, gradientTensor)

		local maximumGradientValue = AqwamTensorLibrary:findMaximumValue(absoluteGradientTensor)

		local mTensor = AqwamTensorLibrary:divide(1, maximumGradientValue)

		local weirdLTensorPart1 = AqwamTensorLibrary:divide(gradientTensor, mTensor)

		local weirdLTensorPart2 = AqwamTensorLibrary:power(weirdLTensorPart1, 2)

		local weirdLTensorPart3 = AqwamTensorLibrary:add(1, weirdLTensorPart2)

		local weirdLTensor = AqwamTensorLibrary:divide(gradientTensor, weirdLTensorPart3)

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(meanMovingAverage, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply((1 - meanMovingAverage), weirdLTensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		firstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, velocityTensor)

		optimizerInternalParameterArray[1] = velocityTensor 
		
		optimizerInternalParameterArray[2] = timeValue

		return firstDerivativeTensor

	end

	return Optimizer.new({CalculateFunction, LearningRateValueScheduler, optimizerInternalParameterArray})

end

function Optimizer.Momentum(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local decayRate = parameterDictionary.decayRate or parameterDictionary[1] or defaultDecayRate

	local weightDecayRate = NewMomentumOptimizer.weightDecayRate or parameterDictionary[2] or defaultWeightDecayRate

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[3]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[4] or {}

	local CalculateFunction = function(learningRate, firstDerivativeTensor, tensor)

		local previousVelocityTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0)

		local gradientTensor = firstDerivativeTensor

		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, tensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(decayRate, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply(learningRate, gradientTensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		firstDerivativeTensor = velocityTensor

		optimizerInternalParameterArray[1] = velocityTensor

		return firstDerivativeTensor

	end

	return Optimizer.new({CalculateFunction, LearningRateValueScheduler, optimizerInternalParameterArray})

end

function Optimizer.NesterovAcceleratedAdaptiveMomentEstimation(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local beta1 = parameterDictionary.beta1 or parameterDictionary[1] or defaultBeta1

	local beta2 = parameterDictionary.beta2 or parameterDictionary[2] or defaultBeta2

	local weightDecayRate = parameterDictionary.weightDecayRate or parameterDictionary[3] or defaultWeightDecayRate

	local epsilon = parameterDictionary.epsilon or parameterDictionary[4] or defaultEpsilon

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[5]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[6] or {}

	local CalculateFunction = function(learningRate, firstDerivativeTensor, tensor)

		local previousMTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0)

		local previousNTensor = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0)

		local timeValue = (optimizerInternalParameterArray[3] or 0) + 1

		local gradientTensor = firstDerivativeTensor

		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, tensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end

		local oneMinusBeta1 = (1 - beta1)

		local meanCostFunctionDerivativeTensor = AqwamTensorLibrary:divide(gradientTensor, oneMinusBeta1)

		local mTensorPart1 = AqwamTensorLibrary:multiply(beta1, previousMTensor)

		local mTensorPart2 = AqwamTensorLibrary:multiply(oneMinusBeta1, gradientTensor)

		local mTensor = AqwamTensorLibrary:add(mTensorPart1, mTensorPart2)

		local meanMTensor = AqwamTensorLibrary:divide(mTensor, oneMinusBeta1)

		local squaredGradientDerivativeTensor = AqwamTensorLibrary:power(gradientTensor, 2)

		local nTensorPart1 = AqwamTensorLibrary:multiply(beta2, previousNTensor)

		local nTensorPart2 = AqwamTensorLibrary:multiply((1 - beta2), squaredGradientDerivativeTensor)

		local nTensor = AqwamTensorLibrary:add(nTensorPart1, nTensorPart2)

		local multipliedNTensor = AqwamTensorLibrary:multiply(beta2, nTensor)

		local meanNTensor = AqwamTensorLibrary:divide(multipliedNTensor, (1 - math.pow(beta2, timeValue)))

		local finalMTensorPart1 = AqwamTensorLibrary:multiply(oneMinusBeta1, meanCostFunctionDerivativeTensor)

		local finalMTensorPart2 = AqwamTensorLibrary:multiply(beta1, meanMTensor)

		local finalMTensor = AqwamTensorLibrary:add(finalMTensorPart1, finalMTensorPart2)

		local squareRootedDivisor = AqwamTensorLibrary:applyFunction(math.sqrt, meanNTensor)

		local finalDivisor = AqwamTensorLibrary:add(squareRootedDivisor, epsilon)

		local firstDerivativeTensorart1 = AqwamTensorLibrary:divide(finalMTensor, finalDivisor)

		firstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, firstDerivativeTensorart1)

		optimizerInternalParameterArray[1] = mTensor
		
		optimizerInternalParameterArray[2] = nTensor
		
		optimizerInternalParameterArray[3] = timeValue

		return tensor

	end

	return Optimizer.new({CalculateFunction, LearningRateValueScheduler, optimizerInternalParameterArray})

end

function Optimizer.RectifiedAdaptiveMomentEstimation(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local beta1 = parameterDictionary.beta1 or parameterDictionary[1] or defaultBeta1
	
	local beta2 = parameterDictionary.beta2 or parameterDictionary[2] or defaultBeta2

	local weightDecayRate = parameterDictionary.weightDecayRate or parameterDictionary[3] or defaultWeightDecayRate

	local epsilon = parameterDictionary.epsilon or parameterDictionary[4] or defaultEpsilon

	local LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler or parameterDictionary[5]

	local optimizerInternalParameterArray = parameterDictionary.optimizerInternalParameterArray or parameterDictionary[6] or {}
	
	local pInfinity = ((2 / (1 - beta2)) - 1)

	local CalculateFunction = function(learningRate, firstDerivativeTensor, tensor)

		local previousMomentumTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0)

		local previousVelocityTensor = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor), 0)

		local timeValue = (optimizerInternalParameterArray[3] or 0) and 1

		local gradientTensor = firstDerivativeTensor

		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, tensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end

		local momentumTensorPart1 = AqwamTensorLibrary:multiply(beta1, previousMomentumTensor)

		local momentumTensorPart2 = AqwamTensorLibrary:multiply((1 - beta1), gradientTensor)

		local momentumTensor = AqwamTensorLibrary:add(momentumTensorPart1, momentumTensorPart2)

		local squaredGradientDerivativeTensor = AqwamTensorLibrary:power(gradientTensor, 2)

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(beta2, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply((1 - beta2), squaredGradientDerivativeTensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		local meanMomentumTensor = AqwamTensorLibrary:divide(momentumTensor, (1 - math.pow(beta1, timeValue)))

		local powerBeta2 = math.pow(beta2, timeValue)

		local p = pInfinity - ((2 * timeValue * powerBeta2) / (1 - powerBeta2))

		if (p > 4) then

			local squareRootVelocityTensor = AqwamTensorLibrary:applyFunction(math.sqrt, velocityTensor)

			local adaptiveLearningRateTensorPart1 = AqwamTensorLibrary:add(squareRootVelocityTensor, NewRectifiedAdaptiveMomentEstimationOptimizer.epsilon)

			local adaptiveLearningRateTensor = AqwamTensorLibrary:divide((1 - powerBeta2), adaptiveLearningRateTensorPart1)

			local varianceRectificationNominatorValue = (p - 4) * (p - 2) * pInfinity

			local varianceRectificationDenominatorValue = (pInfinity - 4) * (pInfinity - 2) * p

			local varianceRectificationValue =  math.sqrt(varianceRectificationNominatorValue / varianceRectificationDenominatorValue)

			firstDerivativeTensor = AqwamTensorLibrary:multiply((learningRate * varianceRectificationValue), meanMomentumTensor, adaptiveLearningRateTensor)

		else

			firstDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, meanMomentumTensor)

		end

		optimizerInternalParameterArray = {momentumTensor, velocityTensor, timeValue}

		return firstDerivativeTensor

	end

	return Optimizer.new({CalculateFunction, LearningRateValueScheduler, optimizerInternalParameterArray})

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

	return Optimizer.new({CalculateFunction, LearningRateValueScheduler, optimizerInternalParameterArray})

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
