"""LoRA SFT of Qwen3.5-4B (shared agent + normalizer) on build_agent_sft.py records, PEFT + transformers Trainer.

Prompt rendering is exactly the inference one (llm_qwen.Qwen.generate): system + user rendered with
add_generation_prompt=True, enable_thinking=False and tokenized on their own; the target is the assistant JSON
followed by <|im_end|>. Loss is on the target tokens only. Records longer than --max-length are skipped (never
truncated: a cut payload or target would teach the wrong thing); the competition context limit is 16K.

LoRA targets the language model only, in all three block types of Qwen3.5: full attention (q/k/v/o_proj), Gated
DeltaNet linear attention (in_proj_qkv, in_proj_z, out_proj) and the MLP; the vision tower and MTP head are left
alone. After training, merge_qwen_lora.py folds the adapter into a bf16 checkpoint with the same parameter count.

  python train_qwen_lora.py --model-path models/Qwen3.5-4B --train outputs/agent_sft/train.jsonl \
      --train outputs/qwen_sft/normalize/train.jsonl --valid outputs/agent_sft/validation.jsonl --out-dir outputs/qwen_lora
For a GPU under ~24 GB add --load-in-4bit (QLoRA; the merge still goes into the bf16 weights).
Install flash-linear-attention and causal-conv1d for fast Gated DeltaNet kernels.
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

PROJECTIONS = {
    'attention': r'self_attn\.(q_proj|k_proj|v_proj|o_proj)',
    'linear_attention': r'linear_attn\.(in_proj_qkv|in_proj_z|out_proj)',
    'mlp': r'mlp\.(gate_proj|up_proj|down_proj)',
}
END_OF_TURN = '<|im_end|>'


def target_regex(groups=('attention', 'linear_attention', 'mlp')):
    """PEFT full-match regex: language-model decoder layers only (no vision tower, no MTP layers)."""
    return r'(.*\.)?language_model\.layers\.\d+\.(' + '|'.join(PROJECTIONS[g] for g in groups) + ')'


def render_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode(tokenizer, record, max_length):
    """input_ids/labels for one chat record; None if it does not fit."""
    messages = record['messages']
    if messages[-1]['role'] != 'assistant':
        raise ValueError(f"{record.get('id')}: last message must be the assistant target")
    prompt_ids = tokenizer(render_prompt(tokenizer, messages[:-1]), add_special_tokens=False)['input_ids']
    target_ids = tokenizer(messages[-1]['content'] + END_OF_TURN, add_special_tokens=False)['input_ids']
    if len(prompt_ids) + len(target_ids) > max_length:
        return None
    return {'input_ids': prompt_ids + target_ids, 'labels': [-100] * len(prompt_ids) + target_ids}


def read_records(paths, tasks=None):
    records = []
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                if tasks is None or record.get('task') in tasks:
                    records.append(record)
    return records


def resample(records, weights, seed):
    """Per-task weight: <1 subsamples, >1 repeats (e.g. normalize=0.5 when it outnumbers agent steps)."""
    rng = random.Random(seed)
    out = []
    for task in sorted({r.get('task') for r in records}):
        group = [r for r in records if r.get('task') == task]
        weight = weights.get(task, 1.0)
        whole, fraction = int(weight), weight - int(weight)
        out += group * whole + rng.sample(group, round(len(group) * fraction))
    rng.shuffle(out)
    return out


class Encoded:
    def __init__(self, tokenizer, records, max_length):
        self.items, self.skipped = [], Counter()
        for record in records:
            item = encode(tokenizer, record, max_length)
            if item is None:
                self.skipped[record.get('task')] += 1
            else:
                self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(pad_id):
    def fn(batch):
        import torch
        width = max(len(item['input_ids']) for item in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, item in enumerate(batch):
            n = len(item['input_ids'])
            ids[row, :n] = torch.tensor(item['input_ids'])
            labels[row, :n] = torch.tensor(item['labels'])
            mask[row, :n] = 1
        return {'input_ids': ids, 'labels': labels, 'attention_mask': mask}
    return fn


def training_kwargs(cls, **kwargs):
    """Same settings on transformers 4.57 and 5.x: v5 dropped warmup_ratio (warmup_steps takes a fraction) and group_by_length."""
    import inspect
    accepted = set(inspect.signature(cls.__init__).parameters)
    if 'warmup_ratio' not in accepted:
        kwargs['warmup_steps'] = float(kwargs.pop('warmup_ratio'))
    if 'eval_strategy' not in accepted and 'evaluation_strategy' in accepted:
        kwargs['evaluation_strategy'] = kwargs.pop('eval_strategy')
    return {key: value for key, value in kwargs.items() if key in accepted}


def generation_check(model, tokenizer, records, max_new_tokens=512):
    """Greedy decode held-out records and compare parsed JSON with the target (what the pipeline actually checks)."""
    import torch
    model.eval()
    result = Counter()
    for record in records:
        prompt = render_prompt(tokenizer, record['messages'][:-1])
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        task = record.get('task')
        result[f'{task}_total'] += 1
        try:
            result[f'{task}_exact_json'] += json.loads(text) == json.loads(record['messages'][-1]['content'])
        except json.JSONDecodeError:
            result[f'{task}_invalid_json'] += 1
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-path', default='models/Qwen3.5-4B')
    parser.add_argument('--train', action='append', required=True)
    parser.add_argument('--valid', action='append', default=[])
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--tasks', default='agent,normalize')
    parser.add_argument('--task-weights', default='', help='e.g. normalize=0.5,agent=1')
    parser.add_argument('--max-length', type=int, default=16384)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--lora-groups', default='attention,linear_attention,mlp')
    parser.add_argument('--epochs', type=float, default=2.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum', type=int, default=16)
    parser.add_argument('--warmup-ratio', type=float, default=0.03)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--no-gradient-checkpointing', action='store_true')
    parser.add_argument('--eval-steps', type=int, default=200)
    parser.add_argument('--save-steps', type=int, default=200)
    parser.add_argument('--generate-eval', type=int, default=64, help='held-out records to greedy-decode at the end (0 = off)')
    parser.add_argument('--limit', type=int, help='first N train records (smoke tests)')
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = set(args.tasks.split(','))
    weights = {k: float(v) for k, v in (item.split('=') for item in args.task_weights.split(',') if item)}
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    train_records = resample(read_records(args.train, tasks), weights, args.seed)[:args.limit]
    valid_records = read_records(args.valid, tasks)
    train_set, valid_set = Encoded(tokenizer, train_records, args.max_length), Encoded(tokenizer, valid_records, args.max_length)
    stats = {'train_records': len(train_set), 'train_skipped_too_long': dict(train_set.skipped), 'valid_records': len(valid_set),
             'train_tasks': dict(Counter(r.get('task') for r in train_records)),
             'max_tokens': max((len(i['input_ids']) for i in train_set.items), default=0),
             'target_tokens': sum(sum(1 for t in i['labels'] if t != -100) for i in train_set.items)}
    print(json.dumps(stats, ensure_ascii=False), flush=True)

    load = {'dtype': torch.bfloat16, 'device_map': args.device_map}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                                         bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **load)
    model.config.use_cache = False
    checkpointing = not args.no_gradient_checkpointing
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing,
                                                gradient_checkpointing_kwargs={'use_reentrant': False})
    elif checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias='none',
                      target_modules=target_regex(tuple(args.lora_groups.split(','))), task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    targeted = sorted({re.sub(r'\.\d+\.', '.N.', n) for n, m in model.named_modules() if hasattr(m, 'lora_A') and n})
    print(json.dumps({'lora_modules': targeted}, ensure_ascii=False), flush=True)
    model.print_trainable_parameters()

    training = TrainingArguments(**training_kwargs(
        TrainingArguments, output_dir=str(out / 'checkpoints'), per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, num_train_epochs=args.epochs,
        lr_scheduler_type='cosine', warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, bf16=True,
        logging_steps=10, save_steps=args.save_steps, save_total_limit=3, eval_strategy='steps' if len(valid_set) else 'no',
        eval_steps=args.eval_steps, load_best_model_at_end=bool(len(valid_set)), metric_for_best_model='eval_loss',
        greater_is_better=False, report_to='none', remove_unused_columns=False, group_by_length=True, seed=args.seed,
        dataloader_num_workers=2, gradient_checkpointing=False, use_cpu=not torch.cuda.is_available()))
    trainer = Trainer(model=model, args=training, train_dataset=train_set, eval_dataset=valid_set if len(valid_set) else None,
                      data_collator=collate(tokenizer.pad_token_id))
    trainer.train()
    adapter = out / 'adapter'
    model.save_pretrained(adapter)
    processor.save_pretrained(adapter)
    summary = {**stats, 'base_model': args.model_path, 'lora': {'r': args.lora_r, 'alpha': args.lora_alpha, 'dropout': args.lora_dropout,
               'groups': args.lora_groups}, 'log_history': trainer.state.log_history[-5:]}
    if args.generate_eval and valid_records:
        model.config.use_cache = True
        summary['generation_check'] = generation_check(model, tokenizer, random.Random(args.seed).sample(valid_records, min(args.generate_eval, len(valid_records))))
    (out / 'train_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != 'log_history'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
