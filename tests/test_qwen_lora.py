import json
import re
import unittest
from pathlib import Path

from train_qwen_lora import encode, render_prompt, resample, target_regex, training_kwargs

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / 'model_metadata' / 'Qwen3.5-4B'

# Module names of Qwen3_5ForConditionalGeneration (transformers), as PEFT sees them.
MODULES = [
    'model.language_model.layers.0.linear_attn.in_proj_qkv', 'model.language_model.layers.0.linear_attn.in_proj_z',
    'model.language_model.layers.0.linear_attn.in_proj_a', 'model.language_model.layers.0.linear_attn.in_proj_b',
    'model.language_model.layers.0.linear_attn.out_proj', 'model.language_model.layers.3.self_attn.q_proj',
    'model.language_model.layers.3.self_attn.o_proj', 'model.language_model.layers.31.mlp.down_proj',
    'model.visual.blocks.0.attn.qkv', 'model.visual.blocks.0.mlp.linear_fc1', 'model.visual.merger.linear_fc2',
    'mtp.layers.0.self_attn.q_proj', 'mtp.layers.0.mlp.up_proj', 'lm_head',
]

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


class TargetTests(unittest.TestCase):
    def test_language_model_projections_only(self):
        pattern = re.compile(target_regex())
        chosen = [m for m in MODULES if pattern.fullmatch(m)]
        self.assertEqual(chosen, [
            'model.language_model.layers.0.linear_attn.in_proj_qkv', 'model.language_model.layers.0.linear_attn.in_proj_z',
            'model.language_model.layers.0.linear_attn.out_proj', 'model.language_model.layers.3.self_attn.q_proj',
            'model.language_model.layers.3.self_attn.o_proj', 'model.language_model.layers.31.mlp.down_proj'])
        self.assertTrue(re.fullmatch(target_regex(), 'base_model.model.model.language_model.layers.2.mlp.gate_proj'))
        attention_only = re.compile(target_regex(('attention',)))
        self.assertFalse(attention_only.fullmatch('model.language_model.layers.0.mlp.down_proj'))

    def test_resample_weights(self):
        records = [{'task': 'agent', 'id': i} for i in range(4)] + [{'task': 'normalize', 'id': i} for i in range(10)]
        out = resample(records, {'normalize': 0.5, 'agent': 2}, 0)
        self.assertEqual(sum(r['task'] == 'agent' for r in out), 8)
        self.assertEqual(sum(r['task'] == 'normalize' for r in out), 5)

    def test_training_kwargs_v5_and_v4(self):
        class V5:
            def __init__(self, warmup_steps=0, eval_strategy='no', seed=0):
                pass
        class V4:
            def __init__(self, warmup_ratio=0.0, evaluation_strategy='no', group_by_length=False):
                pass
        self.assertEqual(training_kwargs(V5, warmup_ratio=0.03, eval_strategy='steps', group_by_length=True, seed=1),
                         {'warmup_steps': 0.03, 'eval_strategy': 'steps', 'seed': 1})
        self.assertEqual(training_kwargs(V4, warmup_ratio=0.03, eval_strategy='steps', group_by_length=True),
                         {'warmup_ratio': 0.03, 'evaluation_strategy': 'steps', 'group_by_length': True})


@unittest.skipIf(AutoTokenizer is None, 'transformers not installed')
class EncodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(QWEN)

    def record(self, target='{"normalized_text": "Bật đèn."}'):
        return {'id': 'r', 'task': 'normalize', 'messages': [
            {'role': 'system', 'content': 'Hệ thống chuẩn hóa.'},
            {'role': 'user', 'content': json.dumps({'transcripts': {'gipformer': ' BẬT ĐÈN'}}, ensure_ascii=False)},
            {'role': 'assistant', 'content': target}]}

    def test_prompt_matches_inference_and_loss_only_on_target(self):
        record = self.record()
        item = encode(self.tokenizer, record, 4096)
        prompt = render_prompt(self.tokenizer, record['messages'][:-1])
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))  # enable_thinking=False, as llm_qwen
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        self.assertEqual(item['input_ids'][:len(prompt_ids)], prompt_ids)
        self.assertTrue(all(label == -100 for label in item['labels'][:len(prompt_ids)]))
        target = [t for t in item['labels'] if t != -100]
        self.assertEqual(target[-1], self.tokenizer.convert_tokens_to_ids('<|im_end|>'))
        self.assertEqual(self.tokenizer.decode(target[:-1]), record['messages'][-1]['content'])

    def test_too_long_is_skipped_not_truncated(self):
        self.assertIsNone(encode(self.tokenizer, self.record(), 10))


if __name__ == '__main__':
    unittest.main()
