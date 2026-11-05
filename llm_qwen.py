import json
import os
import time

class Qwen:
    def __init__(self, trace):
        if os.environ['LLM_BACKEND'] != 'transformers' or os.environ['LLM_QUANTIZATION'] != 'none':
            raise ValueError('Chỉ hỗ trợ backend transformers và quantization none')
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.torch = torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            path = os.environ['LLM_MODEL_PATH']
            self.processor = AutoProcessor.from_pretrained(path)
            load_options = {'dtype': getattr(torch, os.environ['LLM_DTYPE']), 'device_map': os.environ['LLM_DEVICE_MAP']}
            if os.environ.get('LLM_GPU_MAX_MEMORY'):
                load_options['max_memory'] = {0: os.environ['LLM_GPU_MAX_MEMORY'], 'cpu': os.environ.get('LLM_CPU_MAX_MEMORY', '24GiB')}
            self.model = AutoModelForImageTextToText.from_pretrained(path, **load_options)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(f'Không load được Qwen {os.environ["LLM_MODEL_PATH"]}: {exc}. Cần Transformers hỗ trợ Qwen3.5, checkpoint và bộ nhớ phù hợp.') from exc
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.trace = trace
        if self.torch.cuda.is_available():
            free, total = self.torch.cuda.mem_get_info()
            self.trace(stage='llm_loaded', model=path, parameters=self.parameter_count, cuda_allocated_bytes=self.torch.cuda.memory_allocated(), cuda_peak_bytes=self.torch.cuda.max_memory_allocated(), cuda_free_bytes=free, cuda_total_bytes=total)

    def generate(self, system: str, payload: dict, max_tokens: int, stage: str) -> str:
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[prompt], return_tensors='pt').to(self.model.device)
        limit = min(int(os.environ['LLM_CONTEXT_TOKENS']), 16384)
        if inputs['input_ids'].shape[-1] + max_tokens > limit:
            raise ValueError(f'Prompt + output reserve vượt giới hạn {limit} tokens')
        start = time.monotonic()
        kwargs = {'max_new_tokens': max_tokens, 'do_sample': float(os.environ['LLM_TEMPERATURE']) > 0}
        if kwargs['do_sample']:
            kwargs['temperature'] = float(os.environ['LLM_TEMPERATURE'])
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        raw = self.processor.batch_decode(output[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0].strip()
        memory = {}
        if self.torch.cuda.is_available():
            free, total = self.torch.cuda.mem_get_info()
            memory = {'cuda_allocated_bytes': self.torch.cuda.memory_allocated(), 'cuda_peak_bytes': self.torch.cuda.max_memory_allocated(), 'cuda_free_bytes': free, 'cuda_total_bytes': total}
        self.trace(stage=stage, model=os.environ['LLM_MODEL_PATH'], prompt=prompt, input=payload, output=raw, latency=time.monotonic()-start, **memory)
        return raw


def check_competition(llm, trace):
    """Conservatively count checkpoint tensors; fail closed if not inspectable."""
    import sys
    try:
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForSpeechSeq2Seq
        from asr import model_spec
        primary = os.environ['ASR_PRIMARY']
        models = [(primary, model_spec(primary)[1])]
        if os.environ['ASR_MODE'] == 'dual':
            secondary = os.environ['ASR_SECONDARY']
            models.append((secondary, model_spec(secondary)[1]))
        counts = {}
        for kind, path in models:
            if path in counts: continue
            if kind == 'gipformer':
                checkpoint = str(Path(path) / 'model.pt') if Path(path).is_dir() else hf_hub_download(path, 'model.pt')
                data = llm.torch.load(checkpoint, map_location='cpu', weights_only=True)
                state = data['model']
                # State dict also includes buffers: conservative upper bound.
                counts[path] = sum(t.numel() for t in state.values() if isinstance(t, llm.torch.Tensor))
                del data, state
            elif kind == 'zipformer':
                checkpoint = Path(path) / 'jit_script.pt'
                model = llm.torch.jit.load(str(checkpoint), map_location='cpu')
                counts[path] = sum(t.numel() for t in model.state_dict().values())
                del model
            elif kind in ('phowhisper', 'phowhisper_large'):
                config = AutoConfig.from_pretrained(path)
                with init_empty_weights():
                    model = AutoModelForSpeechSeq2Seq.from_config(config)
                counts[path] = sum(p.numel() for p in model.parameters())
                del model
            else: raise ValueError(f'ASR không hỗ trợ: {kind}')
        total = llm.parameter_count + sum(counts.values())
        over = llm.parameter_count >= 5_000_000_000 or total >= 6_000_000_000 or int(os.environ['LLM_CONTEXT_TOKENS']) > 16384
        trace(stage='competition_limits', llm_parameters=llm.parameter_count, asr_parameters_upper_bound=counts, total_parameters_upper_bound=total, over=over)
        print(f"{'OVER COMPETITION LIMIT' if over else 'COMPETITION PARAMETERS'}: LLM={llm.parameter_count}, ASR={counts}, total<={total}", file=sys.stderr)
        if over and os.environ['STRICT_COMPETITION'] == '1':
            raise ValueError('OVER COMPETITION LIMIT: vượt giới hạn tham số')
        return not over
    except Exception as exc:
        trace(stage='competition_limits_error', error=str(exc))
        if os.environ['STRICT_COMPETITION'] == '1':
            raise RuntimeError(f'Không xác minh được giới hạn tham số: {exc}') from exc
        print(f'OVER COMPETITION LIMIT / KHÔNG XÁC MINH ĐƯỢC: {exc}', file=sys.stderr)
        return False
