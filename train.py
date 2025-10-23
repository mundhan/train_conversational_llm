import os
import glob

import wandb
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

EPOCH = 3
DTYPE = "bf16"
PDTBS = 1
GAS = 128
DATASET = "###"
LR = 2e-4
RANK = 32
LORA_ALPHA = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### 1. WandB 설정
wandb.login()
wandb_project = f"qwen3_4b_{DATASET}_{PDTBS}_{GAS}_{DTYPE}_e{EPOCH}_lr{LR}_R{RANK}_LA{LORA_ALPHA}"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

# 2. 모델 및 토크나이저 불러오기
base_model = "###"
tokenizer = AutoTokenizer.from_pretrained(base_model, add_bos=False)
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = "right"

# 3. 모델 로드 및 LoRA 설정
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
model.config.use_cache = False
model.gradient_checkpointing_disable()

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# LoRA 설정 추가
lora_config = LoraConfig(
    r=RANK,
    lora_alpha=LORA_ALPHA,     
    lora_dropout=0.05, 
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    exclude_modules=["vision_tower"],
)
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()  # trainable params > 0 이어야 정상
n_lora = sum(p.numel() for n,p in model.named_parameters() if "lora_" in n)
print("n_lora_params:", n_lora)  # 0이면 주입 실패
assert n_lora > 0, "LoRA injection failed: no lora_* params created"

# 4. 데이터 로딩 및 전처리 (dev 제외)
data_path = "###"
raw_dataset = load_dataset("json", data_files={"train": data_path})["train"]

def format(example):
    messages = [
        {
            "role": "system",
            "content": f"""###"""}
    ]
    for turn in example.get("history", []):
        messages.append({
            "role": turn.get("role", "user"),
            "content": turn.get("content", ""),
        })
    resp = example.get("response", {})
    messages.append({
        "role": resp.get("role", "assistant"),
        "content": resp.get("content", ""),
    })
    
    # 전체 대화 문자열화 (마지막 assistant까지 포함)
    full_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False
    )

    # 마지막 assistant 이전까지만 포함한 부분 (context)
    context_text = tokenizer.apply_chat_template(
        messages[:-1],  # 마지막 assistant 제외
        add_generation_prompt=True,  # assistant turn 시작 토큰만 남김
        tokenize=False
    )

    # 인코딩
    inputs = tokenizer(
        full_text,
        max_length=2048,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # 라벨 복제 후, context 부분은 마스크(-100)
    labels = inputs["input_ids"].clone()

    # context 길이 계산
    context_ids = tokenizer(
        context_text,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    )["input_ids"]

    context_len = context_ids.size(1)
    labels[:, :context_len] = -100  # context 구간 마스크 처리

    inputs["labels"] = labels

    ## 디버깅용 출력
    # print("=== Context ===")
    # print(context_text)
    # print("\n=== Full ===")
    # print(full_text)
    # print(f"\nContext tokens: {context_len}, Total: {inputs['input_ids'].size(1)}")

    return {k: v.squeeze(0) for k, v in inputs.items()}

def get_input_length(example):
    return len(tokenizer(example['input'], truncation=True, max_length=2048)['input_ids'])

tokenized = raw_dataset.shuffle(seed=42).map(format, remove_columns=raw_dataset.column_names)

# 전체 step 수 계산
num_update_steps_per_epoch = len(tokenized) // (PDTBS * GAS)
max_steps = num_update_steps_per_epoch * EPOCH
print('steps: ', int(max_steps * 0.08))

# 5. 학습 설정
training_args = TrainingArguments(
    output_dir=f"###",
    # epoch
    num_train_epochs=EPOCH,
    # 배치 크기
    per_device_train_batch_size=PDTBS,
    gradient_accumulation_steps=GAS,
    # 웜업, 클리핑
    warmup_ratio=0.03,
    max_grad_norm=1.0,  # gradient clipping 적용
    weight_decay=0.0,  # L2 정규화
    # 학습률
    learning_rate=LR,
    lr_scheduler_type="cosine", # "cosine_with_restarts" 또는 "linear"
    # 옵티마이저
    optim="adamw_torch",
    # 모델 저장
    save_strategy="steps",#"steps",
    save_steps=int(max_steps * 0.08),
    # 평가 출력
    logging_steps=int(max_steps * 0.08),
    bf16=True,
    gradient_checkpointing=False,
    # 학습 추적
    report_to="wandb",  # wandb 사용
    run_name=f"{wandb_project}",
    # 데이터셋
    remove_unused_columns=False,
    save_safetensors=True,
    ## 모델 평가
    # eval_strategy="steps",#"steps",
    # eval_steps=int(max_steps * 0.08),
)

# 6. Trainer로 학습 시작
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized,
    # eval_dataset=dev_tokenized,
)

trainer.train() # 체크포인트에서 이어서 학습 시: (resume_from_checkpoint=True)
# 최종 저장: 어댑터만 (명시적)
adapter_out = f"###"
model.save_pretrained(adapter_out, safe_serialization=True)
print(f"[OK] LoRA adapters saved at: {adapter_out}")