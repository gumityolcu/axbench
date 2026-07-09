from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.HF_handlers.HF_dataset_handler import TRUNCATION_MAX_LENGTH
import torch 

class HFModelAdapter:
    def __init__(self, model_name, dtype=torch.bfloat16, device_map={"": "cuda:0"}):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

    def format_chat(self, messages, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def tokenize(self, texts, max_length=TRUNCATION_MAX_LENGTH):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    @torch.inference_mode()
    def forward_hidden(self, texts, max_length=TRUNCATION_MAX_LENGTH):
        inputs = self.tokenize(texts, max_length=max_length)
        return self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        ), inputs