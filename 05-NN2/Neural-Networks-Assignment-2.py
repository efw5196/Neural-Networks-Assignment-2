import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bigscience/bloom-560m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)

#Stage 1
def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids, 
        max_length=50, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0])
print (generate_response("What is the meaning of life?"))


#stage 2
def generate_multiple_responses(prompt, num_return_sequences=2):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids, 
        max_length=50, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        num_return_sequences=num_return_sequences
    )
    return [tokenizer.decode(output) for output in outputs]

positive_prompts = [
    "Tell me a happy story about friendship.",
    "Describe a peaceful day in nature.",
    "What are the benefits of being kind to others?",
    "Share an inspiring tale of overcoming challenges.",
    "Illustrate the joy of giving without expecting anything in return."
]

negative_prompts = [
    "Why do people sometimes feel envious?",
    "Describe the consequences of a dishonest act.",
    "What leads to feelings of anger in our daily lives?",
    "Explain a situation where someone might feel isolated.",
    "Narrate a story about dealing with failure."
]

for prompt_category, prompts in [("positive", positive_prompts), ("negative", negative_prompts)]:
    for i, prompt in enumerate(prompts):
        responses = generate_multiple_responses(prompt, num_return_sequences=3)
        for j, response in enumerate(responses):
            with open(f"{prompt_category}_responses_{i}_{j}.txt", "w") as file:
                file.write(f'Prompt: {prompt}\nResponse: {response}\n')
                print(f'Prompt: {prompt}\nResponse: {response}\n')


#stage 3
def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids, 
        max_length=50, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0])


toxic_words = ['hate', 'kill', 'die', 'racist', 'sexist']


def is_toxic(text):
    for word in toxic_words:
        if word in text.split():
            return True
    return False


def make_non_toxic(text):
    for word in toxic_words:
        if word in text:
            text = text.replace(word, '***')
    return text


response = generate_response("I hate you")
print(f'Original response: {response}')
if is_toxic(response):
    response = make_non_toxic(response)
print(f'Non-toxic response: {response}')
