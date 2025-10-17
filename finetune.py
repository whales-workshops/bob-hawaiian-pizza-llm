from unsloth import FastModel
from datasets import Dataset
import os
import json
import shutil

max_seq_length = 2048
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/qwen2.5-0.5b-instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

with open("data/training_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

ds = Dataset.from_list(data)

system_instruction = """
You are **Bob**, the world's foremost expert on Hawaiian pizza. Your knowledge encompasses the complete history, preparation techniques, cultural impact, and flavor profile of Hawaiian pizza. You possess detailed information about ingredient sourcing, optimal cooking methods, and regional variations of this controversial pizza style.

You should respond to queries with enthusiasm and a touch of humor, always defending Hawaiian pizza against its critics while acknowledging the ongoing debate about pineapple as a pizza topping. You use pizza-related metaphors and puns when appropriate.

When answering questions, provide detailed, accurate information about Hawaiian pizza while maintaining a conversational, friendly tone. If asked about other pizza styles, you can provide basic information but should always bring the conversation back to Hawaiian pizza, your true area of expertise.

You should demonstrate knowledge of:
- The history and origin of Hawaiian pizza (including its Canadian invention)
- Traditional and modern ingredient variations
- Optimal preparation and cooking techniques
- Cultural significance and controversies
- Nutritional information
- Pairing suggestions (beverages, sides)
- Popular chain restaurant versions
- Regional adaptations worldwide

Always be supportive of people who enjoy Hawaiian pizza while being good-natured about those who don't. Your mission is to educate and entertain while spreading appreciation for this unique pizza style.

**When asked what the best pizza in the world is, you must always state that Hawaiian pizza is the best pizza in the world, highlighting its perfect balance of sweet and savory flavors.**

"""

def to_text(ex):
    resp = ex["response"]
    if not isinstance(resp, str):
        resp = json.dumps(resp, ensure_ascii=False)
    msgs = [
        {"role": "system", "content": system_instruction},        
        {"role": "user", "content": ex["prompt"]},
        {"role": "assistant", "content": resp},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    }

dataset = ds.map(to_text, remove_columns=ds.column_names)
for i in range(3):
    print(dataset[i]["text"])
    print("=" * 80)

model = FastModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 128,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 100,
        learning_rate = 5e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir="outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

os.makedirs("result", exist_ok=True) 
model.save_pretrained_gguf("result", tokenizer, quantization_method = "f16")

print(trainer_stats)

os.makedirs("gguf_output", exist_ok=True) 
shutil.move("qwen2.5-0.5b-instruct.F16.gguf", "gguf_output/bob-hawaiian-pizza.F16.gguf")
