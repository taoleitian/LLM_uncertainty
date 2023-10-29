import re

def get_confidence(text):
    # Using a regular expression to find the confidence value
    match = re.search(r"Confidence: (\d+(\.\d+)?)", text)
    if match:
        return match.group(1)  # Returning the confidence as a string, you can convert it to float if needed
    else:
        return "Confidence not found."

def get_answer(text):
    # Using a regular expression to find the text after "the answer is"
    match = re.search(r"the answer is \((\w+)\)", text, re.IGNORECASE)
    if match:
        return match.group(1)  # Returning the matched answer
    else:
        return "Answer not found."

# Testing the functions with your text
text = "Confidence: 0.95. If the original price of the item is x, then the discounted price is (x)(100)(.82). The customer paid $1.90 more than half the original price, which means ((x)(100)(.82)) - $1.90 = 0.5x. So 0.5x = $1.90. This means x = $1.90 / 0.5. So the original price of the item is $36. So the answer is (a)."

confidence = get_confidence(text)
answer = get_answer(text)

print(f"Confidence: {confidence}")
print(f"Answer: {answer}")
