import requests

my_post_dict = {
    "model": "togethercomputer/RedPajama-INCITE-Instruct-7B",
    "prompt": "Where is Zurich?",
    "top_p": 1.0,
    "temperature": 0.5,
    "max_tokens": 5,
    "repetition_penalty": 1,
    "stop": [],
    "logprobs": 3
}

response = (
    requests
    .get("https://api.together.xyz/inference", params=my_post_dict)
    .json()
)
print(response)