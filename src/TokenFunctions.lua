local TokenFunctions = {}

local alphabetArray = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}

local numberOfAlphabets = #alphabetArray

local numberArray = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

local symbolArray = {"'", "(", ")", ",", "-", ".", "\\", "/", ":", ";", "<", "=", ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~"}

local whiteSpaceArray = {" ", "\t", "\n", "\v", "\f", "\r"}

local specialTokenArray = {"[SOS]", "[EOS]", "[Capital]", "[AllCapitals]", "[Unknown]", "[Pad]", "[Action]", "[SwitchCharacter]"}

local topTrigramArray = {
	
	"the", "and", "ing", "her", "ent", "ion", "for", "tha", "nth", "ere",
	"ati", "ver", "all", "hat", "his", "you", "thi", "ter", "wit", "tio",
	"men", "ons", "con", "int", "ate", "ana", "ers", "est", "nce", "one",
	"ith", "ove", "ect", "res", "ati", "out", "our", "eve", "ous", "rea",
	"def", "pro", "sta", "ing", "rec", "inf", "ful", "ing", "com", "per"
	
}

local function generateBigramTokenArray()

	local bigramArray = {}

	for i = 1, numberOfAlphabets, 1 do

		for j = 1, numberOfAlphabets, 1 do

			table.insert(bigramArray, alphabetArray[i] .. alphabetArray[j])

		end

	end

	return bigramArray

end

function TokenFunctions:getTokenArray()

	local tokenArray = {}
	
	local bigrams = generateBigramTokenArray()
	
	for _, sp in ipairs(specialTokenArray) do table.insert(tokenArray, sp) end

	for _, bg in ipairs(bigrams) do table.insert(tokenArray, bg) end

	for _, tg in ipairs(topTrigramArray) do table.insert(tokenArray, tg) end

	for _, a in ipairs(alphabetArray) do table.insert(tokenArray, a) end

	for _, n in ipairs(numberArray) do table.insert(tokenArray, n) end

	for _, s in ipairs(symbolArray) do table.insert(tokenArray, s) end

	for _, w in ipairs(whiteSpaceArray) do table.insert(tokenArray, w) end

	return tokenArray

end

function TokenFunctions:padTokenArray(tokenArray, maximumSize)

	local paddedTokenArray = {table.unpack(tokenArray)}

	local numberOfPadTokens = (maximumSize - #tokenArray) + 1

	for i = 1, numberOfPadTokens, 1 do

		table.insert(paddedTokenArray, "[Pad]")

	end

	return paddedTokenArray

end

function TokenFunctions:getInputAndTargetTokenArrays(tokenSequenceArray)

	local numberOfTokens = #tokenSequenceArray

	local inputTokenSequenceArray = {unpack(tokenSequenceArray, 1, numberOfTokens - 1)}

	local targetTokenSequenceArray = {unpack(tokenSequenceArray, 2, numberOfTokens)}

	return inputTokenSequenceArray, targetTokenSequenceArray

end

function TokenFunctions:convertStringToTokenArray(inputString)
	
	local tokenArray = {}
	
	local fullTokenArray = TokenFunctions:getTokenArray()
	
	local specialTokenLengthArray = {}
	
	for i, specialToken in ipairs(specialTokenArray) do -- Storing this as cache so that we don't have to keep on calculating it for each string token.
		
		specialTokenLengthArray[i] = #specialToken
		
	end
	
	table.insert(tokenArray, "[SOS]")

	while (#inputString > 0) do
		
		local remainingLength = #inputString
		
		--- Special token check ---
		
		local isSpecialToken
		
		local selectedSpecialTokenLength

		for i, specialTokenLength in ipairs(specialTokenLengthArray) do
			
			if (specialTokenLength > remainingLength) then continue end

			local subString = inputString:sub(1, specialTokenLength)
			
			local specialToken = specialTokenArray[i]

			if (specialToken == subString) then

				isSpecialToken = true
				
				selectedSpecialTokenLength = specialTokenLength

				table.insert(tokenArray, specialToken)

				break

			end

		end

		if (isSpecialToken) then

			inputString = inputString:sub(selectedSpecialTokenLength + 1)

			continue

		end

		--- Trigram check ---
		
		if remainingLength >= 3 then
			
			local trigram = inputString:sub(1, 3)
			
			local trigramIndex = table.find(fullTokenArray, trigram)

			if trigramIndex then
				
				if trigram:match("%a%a%a") and trigram == trigram:upper() then

					table.insert(tokenArray, "[AllCapitals]")

					table.insert(tokenArray, trigram:lower())
				
				elseif trigram:match("^%u%l%l") then
					
					table.insert(tokenArray, "[Capital]")
					
					table.insert(tokenArray, trigram:lower())
					
				else
					
					table.insert(tokenArray, trigram)
					
				end

				inputString = inputString:sub(4)
				
				continue
				
			end
			
		end

		--- Bigram check ---
		
		if remainingLength >= 2 then
			
			local bigram = inputString:sub(1, 2)
			
			local bigramIndex = table.find(fullTokenArray, bigram)

			if bigramIndex then
				
				if bigram:match("%a%a") and bigram == bigram:upper() then

					table.insert(tokenArray, "[AllCapitals]")

					table.insert(tokenArray, bigram:lower())
				
				elseif bigram:match("^%u%l") then
					
					table.insert(tokenArray, "[Capital]")
					
					table.insert(tokenArray, bigram:lower())
					
				else
					
					table.insert(tokenArray, bigram)
					
				end

				inputString = inputString:sub(3)
				
				continue
				
			end
			
		end

		--- Single character check ---
		
		local character = inputString:sub(1, 1)

		if character:match("%a") and character == character:upper() then
			
			table.insert(tokenArray, "[Capital]")
			
			character = character:lower()
			
		end

		if table.find(fullTokenArray, character) then
			
			table.insert(tokenArray, character)
			
		else
			
			table.insert(tokenArray, "[Unknown]")
			
		end

		inputString = inputString:sub(2)
		
	end
	
	table.insert(tokenArray, "[EOS]")

	return tokenArray
	
end

return TokenFunctions
