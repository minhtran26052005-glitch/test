"""GipFormer ONNX transducer inference via sherpa-onnx."""
from pathlib import Path

class GipFormer:
    def __init__(self, model_path: str, device: str = 'cpu'):
        try:
            import sherpa_onnx
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError('GipFormer cần sherpa-onnx và huggingface-hub') from exc
        path = Path(model_path)
        if not path.is_dir():
            try:
                path = Path(snapshot_download(model_path, allow_patterns=['*.onnx', 'tokens.txt']))
            except Exception as exc:
                raise RuntimeError(f'Không tải được checkpoint GipFormer {model_path}: {exc}') from exc
        def find(part: str) -> str:
            files = sorted(p for p in path.rglob(f'{part}*.onnx') if 'int8' not in p.name.lower())
            if len(files) != 1:
                raise RuntimeError(f'Cần đúng một file {part} ONNX không quantize trong {path}; tìm thấy {len(files)}')
            return str(files[0])
        tokens = list(path.rglob('tokens.txt'))
        if len(tokens) != 1:
            raise RuntimeError(f'Không tìm thấy tokens.txt duy nhất trong {path}')
        provider = {'cpu': 'cpu', 'cuda': 'cuda'}.get(device)
        if provider is None:
            raise ValueError('ASR_DEVICE phải là cpu hoặc cuda')
        self.model = sherpa_onnx.OfflineRecognizer.from_transducer(encoder=find('encoder'), decoder=find('decoder'), joiner=find('joiner'), tokens=str(tokens[0]), num_threads=4, sample_rate=16000, feature_dim=80, decoding_method='greedy_search', provider=provider)

    def transcribe(self, audio_path: str) -> str:
        return self.transcribe_detailed(audio_path)['text']

    def transcribe_detailed(self, audio_path: str) -> dict:
        import soundfile as sf
        import numpy as np
        audio, rate = sf.read(audio_path, dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        if rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
        stream = self.model.create_stream()
        stream.accept_waveform(16000, np.ascontiguousarray(audio))
        self.model.decode_stream(stream)
        result = stream.result
        return {'text': str(result.text), 'tokens': list(result.tokens),
                'timestamps': list(result.timestamps), 'ys_log_probs': list(result.ys_log_probs),
                'durations': list(result.durations), 'sample_rate': 16000,
                'duration_seconds': len(audio) / 16000}
