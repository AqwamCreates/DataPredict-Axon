local DisplayErrorFunctions = {}

DisplayErrorFunctions.displayFunctionErrorDueToNonObjectCondition = function(showError) if (showError) then error("This function can only be called if it is an object.") end end

return DisplayErrorFunctions
