
from datasets import load_dataset as HF_load_dataset

TRUNCATION_MAX_LENGTH = 1024
def tiny_stories(split, streaming, tokenizer, **kwargs):
    def collate_fn(batch):
        texts = [x["text"] for x in batch]
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=TRUNCATION_MAX_LENGTH)
    return HF_load_dataset("roneneldan/TinyStories", split=split, streaming=streaming), collate_fn

loader_functions = {
    "roneneldan/TinyStories": tiny_stories
}

def load_dataset(name, config, split, streaming, tokenizer):
    fn=loader_functions.get(name, None)
    if fn is None:
        raise ValueError(f"Dataset not supported: {name}")
    return fn(split=split, config=config, streaming=streaming, tokenizer=tokenizer)