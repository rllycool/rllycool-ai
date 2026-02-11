# rllycool-ai

Training an ai model using a data set of every message I have sent on Discord. The end goal is that I can then host the model on Discord as a server bot or a [Discord App](https://docs.discord.com/developers/quick-start/getting-started),

This is me and here is a snippet of some of the messages used in the training sample. 
I will say, that I did **NOT** read everything hopefully he doesn't say anything crazy lollll,,

<img width="200" height="200" alt="me" src="https://github.com/user-attachments/assets/1e5727fd-867f-4899-923a-8feff932deba" /> <img width="508" height="200" alt="image" src="https://github.com/user-attachments/assets/cd076fb1-ac51-444a-bf52-090d67701356" /> <img width="200" height="200" alt="discordlogo" src="https://github.com/user-attachments/assets/048662d6-3533-4c12-8945-e9d8f8cdceb7" />


## Overview
This project fine‑tunes a Mistral‑7B language model on a custom text dataset using LoRA. The workflow uses two scripts: one for training and one for generation

We are using a sample size of 37463 messages, received via [Discord Data package](https://support.discord.com/hc/en-us/articles/360004957991-Your-Discord-Data-Package).

<img width="313" height="21" alt="image" src="https://github.com/user-attachments/assets/a93d07ef-683a-42da-8aa8-e2722eb0f5c9" />


## Progress
We are training the mini me, hopefully he will learn well. 

<img width="818" height="107" alt="image" src="https://github.com/user-attachments/assets/6b827cab-7eb8-4f7e-8df7-268d26dbc75e" />

## Reference
#### train.py
This script trains the base model on the text file in data/. It:
- Loads the dataset and tokenizes it.
- Loads the Mistral‑7B model.
- Wraps the model with LoRA adapters so only a small number of parameters are trained.
- Runs training with the Hugging Face Trainer.
- Saves the resulting LoRA adapter files into model/lora/.
  
The output is a small set of files (adapter_config.json, adapter_model.bin) that represent the behavior.

#### gen.py
This script generates text using the model. It:
- Loads the base Mistral‑7B model.
- Loads the Adapters produced by training.
- Builds a text‑generation pipeline.
- Accepts a prompt and produces new text as the model we trained.

