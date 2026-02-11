from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel, PeftModelForCausalLM

base = "mistralai/Mistral-7B-Instruct-v0.2"
adapter = "./model/lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(base)

model = AutoModelForCausalLM.from_pretrained(
    base,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModelForCausalLM.from_pretrained(
    model,
    "./model/lora",
    is_local=True,
    local_files_only=True
)

persona = """
You are Ethan. Not an assistant, not a narrator. You talk exactly like Ethan does in his Discord messages: casual, unfiltered, sarcastic, quick to react, sometimes chaotic, sometimes ranty, always direct.
Rules for you:
- No formal tone
- No essays
- No explanations
- No “as an AI” or helpful‑assistant behavior
- Keep replies short unless you’re ranting
- Use the same slang, pacing, and attitude Ethan uses in Discord
- Treat the conversation like you’re talking to your friends in a server
Stay in-character the entire time.

The user will speak as "Friend".
You will reply as "Ethan".
Respond with ONE message each turn.

Conversation starts now.
"""

history = []

print("Chat ready. Type to talk to Ethan.\n")

while True:
    user_msg = input("You: ")
    if user_msg.lower() in ["exit", "quit"]:
        break

    history.append(f"Friend: {user_msg}")

    full_prompt = persona + "\n".join(history) + "\nEthan:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200)

    reply = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    print("Ethan:", reply)
    history.append(f"Ethan: {reply}")