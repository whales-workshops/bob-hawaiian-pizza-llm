# Labspace - Fine tuning
> This Labspace provides tools to fine-tune models using Docker Offload, Docker Model Runner, and Unsloth.

## Start the local development mode:

```bash
# On Mac/Linux
CONTENT_PATH=$PWD docker compose -f oci://dockersamples/labspace-content-dev -f .labspace/compose.override.yaml up

# On Windows with PowerShell
$Env:CONTENT_PATH = (Get-Location).Path; docker compose -f oci://dockersamples/labspace-content-dev -f .labspace/compose.override.yaml up
```

And then open your browser to http://localhost:3030.

### Dataset Source

The training dataset `data/training_data.json` used in this example was created with Claude Code.

### Fine-Tuning Process

The `finetune.py` script demonstrates how to:

1. Load a pre-trained model (`unsloth/qwen2.5-0.5b-instruct`)
2. Prepare the dataset for instruction fine-tuning
3. Configure LoRA adapters for efficient training
4. Train the model using the SFTTrainer from TRL
5. Save the fine-tuned model

```bash
docker offload start
docker compose -f oci://philippecharriere494/bob-hawaiian-pizza-llm up -d
```