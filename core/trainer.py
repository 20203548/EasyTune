import os
import torch
import json
from transformers import TrainerCallback
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


class LossLoggerCallback(TrainerCallback):
    """实时记录loss到JSON文件"""

    def __init__(self, output_file):
        self.output_file = output_file
        self.loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
            }
            self.loss_history.append(log_entry)

            # 实时保存到文件
            with open(self.output_file, 'w') as f:
                json.dump({"log_history": self.loss_history}, f, indent=2)


def train_model(
        task_id,
        data_path,
        user_args,
):
    base_model = user_args.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct")
    # ================= 1. 路径设置 =================
    current_file_path = os.path.abspath(__file__)
    core_dir = os.path.dirname(current_file_path)
    project_root = os.path.dirname(core_dir)
    output_dir = os.path.join(project_root, "output", task_id)
    logging_dir = os.path.join(project_root, "logs", task_id)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    print(f"\n[EasyTune] 🚀 任务启动: {task_id}")

    # ================= 2. 加载 Tokenizer =================
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # 修复 Warning 的关键：如果没有 pad_token，就用 eos_token 顶替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ================= 3. 数据处理 (核心修正!!!) =================
    def process_func(example):
        instruction = example.get('instruction', '')
        response = example.get('output', '')

        # --- 核心修改开始 ---
        # 使用 apply_chat_template 保证和推理时的格式一模一样
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
        # 生成标准的训练文本
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # --- 核心修改结束 ---

        ids = tokenizer(text, padding=False, truncation=True, max_length=512)
        return {
            "input_ids": ids["input_ids"],
            "attention_mask": ids["attention_mask"],
            "labels": ids["input_ids"]
        }

    dataset = load_dataset("json", data_files=data_path, split="train")
    tokenized_ds = dataset.map(process_func)

    # ================= 4. 加载模型 =================
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=user_args.get("lora_r", 8),
        lora_alpha=user_args.get("lora_alpha", 16),
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    # ================= 5. 训练参数 (加强版) =================
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=user_args.get("batch_size", 2),
        gradient_accumulation_steps=user_args.get("gradient_accumulation_steps", 4),
        num_train_epochs=user_args.get("epoch", 1),
        learning_rate=user_args.get("learning_rate", 3e-4),
        logging_steps=1,
        logging_dir=str(logging_dir),
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),
        report_to=None
    )

    loss_callback = LossLoggerCallback(os.path.join(logging_dir, "trainer_state.json"))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[loss_callback]  # 新增：添加自定义回调
    )

    print("[EasyTune] ▶️  开始训练...")
    trainer.train()

    model.save_pretrained(output_dir)
    print(f"[EasyTune] ✅ 训练完成！结果已保存至: {output_dir}")
    return output_dir


if __name__ == "__main__":
    train_model(
        task_id="ewew",
        data_path="../data/45d8e227-8808-4230-a16c-f6be7296d4d5.json",
        user_args = {
        }
    )
