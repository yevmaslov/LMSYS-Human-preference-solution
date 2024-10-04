## Solution for LMSYS - Chatbot Arena Human Preference Predictions 

Task: predict which responses users will prefer in a head-to-head battle between LLMs - response A, response B, or both responses are equally good/bad. 

Summary: The solution consists of two language models
* gemma-2-9b-it finetuned for sequence classification. The objective is straightforward - multi-turn conversation between human and LLM as input, logits as output. 
* llama-3.1-8b-instruct finetuned for text generation. The objective was to make the model generate text, indicating which response is better, and use a serving engine for faster inference (vLLM).

The weighted average of the logits from these two models is taken as the final prediction. 

Training Hardware: 
* RAM: 48GB
* GPU - 1x A6000 Ada (48GB) / A100 (80GB)

Inference Hardware:
* RAM: 30GB
* GPU - 2xT4 (16GB)