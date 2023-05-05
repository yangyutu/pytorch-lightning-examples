from transformers import AutoModelForSeq2SeqLM

# huggingface hub model id
model_id = "google/flan-t5-base"

# load model from the hub
# You can load a model by roughly halving the memory requirements by using load_in_8bit=True argument when calling .from_pretrained method
model_8bit = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")


print(model.get_memory_footprint()) # 990,311,424
print(sum(p.numel() for p in model.parameters())) #247,577,856

print(model_8bit.get_memory_footprint()) # 410,221,056
print(sum(p.numel() for p in model_8bit.parameters())) #247,577,856