"""GipFormer ONNX transducer inference via sherpa-onnx.

Decoding defaults to greedy_search, the decoding every saved ASR cache, the fusion selector and all benchmarks
were made with. `decoding` (see decoding_options) switches one model to modified_beam_search, optionally with
hotwords: a contextual-biasing list of domain words (SJC, ZALO, SMS, ...) that needs no retraining.
"""
import hashlib
import os
import tempfile
from pathlib import Path

DECODING_METHODS = ('greedy_search', 'modified_beam_search')


def decoding_options(prefix, environ=None):
    """sherpa-onnx transducer options from {PREFIX}_DECODING / _BEAM / _HOTWORDS / _HOTWORDS_SCORE.

    Empty when nothing is set (greedy), so cache keys and benchmark identities of earlier runs stay valid.
    Hotwords only act in modified_beam_search (sherpa-onnx ignores them in greedy search), so asking for
    hotwords without beam search is an error rather than a silent no-op.
    """
    environ = os.environ if environ is None else environ
    method = environ.get(f'{prefix}_DECODING') or 'greedy_search'
    hotwords = environ.get(f'{prefix}_HOTWORDS') or ''
    if method not in DECODING_METHODS:
        raise ValueError(f'{prefix}_DECODING must be one of {DECODING_METHODS}, not {method!r}')
    if hotwords and method != 'modified_beam_search':
        raise ValueError(f'{prefix}_HOTWORDS needs {prefix}_DECODING=modified_beam_search')
    for key in ('BEAM', 'HOTWORDS_SCORE'):
        if environ.get(f'{prefix}_{key}') and method != 'modified_beam_search':
            raise ValueError(f'{prefix}_{key} needs {prefix}_DECODING=modified_beam_search')
    if method == 'greedy_search':
        return {}
    options = {'decoding_method': method, 'max_active_paths': int(environ.get(f'{prefix}_BEAM') or 4)}
    if hotwords:
        if not Path(hotwords).is_file():
            raise FileNotFoundError(f'{prefix}_HOTWORDS={hotwords} does not exist')
        options.update(hotwords_file=str(hotwords), hotwords_score=float(environ.get(f'{prefix}_HOTWORDS_SCORE') or 1.5))
    return options


def decoding_identity(options):
    """What the transcript depends on besides the model and audio: the options and the hotword list's content."""
    if not options:
        return None
    identity = dict(options)
    if identity.get('hotwords_file'):
        identity['hotwords_sha256'] = hashlib.sha256(Path(identity.pop('hotwords_file')).read_bytes()).hexdigest()
    return identity


def bpe_vocab(model_dir):
    """bpe.vocab (piece<TAB>score, sherpa-onnx scripts/export_bpe_vocab.py) that sherpa-onnx tokenizes hotwords with.

    Taken from the model directory when present, otherwise exported from its bpe.model into a temp file keyed by
    the model's hash: writing into the model directory would change its fingerprint and invalidate every cache.
    """
    path = Path(model_dir)
    found = sorted(path.rglob('bpe.vocab'))
    if found:
        return str(found[0])
    models = sorted(path.rglob('bpe.model'))
    if not models:
        raise RuntimeError(f'Hotwords need bpe.vocab or bpe.model in {path} (the BPE the model was trained with)')
    data = models[0].read_bytes()
    target = Path(tempfile.gettempdir()) / 'sherpa_bpe_vocab' / f'{hashlib.sha256(data).hexdigest()[:16]}.vocab'
    if not target.is_file():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_proto=data)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(''.join(f'{sp.id_to_piece(i)}\t{sp.get_score(i)}\n' for i in range(sp.get_piece_size())),
                             encoding='utf-8')
        temporary.replace(target)
    return str(target)


class GipFormer:
    def __init__(self, model_path: str, device: str = 'cpu', decoding: dict = None):
        try:
            import sherpa_onnx
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError('GipFormer cần sherpa-onnx và huggingface-hub') from exc
        decoding = dict(decoding or {})
        path = Path(model_path)
        if not path.is_dir():
            patterns = ['*.onnx', 'tokens.txt'] + (['bpe.model', 'bpe.vocab'] if decoding.get('hotwords_file') else [])
            try:
                path = Path(snapshot_download(model_path, allow_patterns=patterns))
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
        if decoding.get('hotwords_file'):
            decoding.update(modeling_unit='bpe', bpe_vocab=bpe_vocab(path))
        options = {'decoding_method': 'greedy_search', **decoding}
        self.decoding = options
        self.model = sherpa_onnx.OfflineRecognizer.from_transducer(encoder=find('encoder'), decoder=find('decoder'), joiner=find('joiner'), tokens=str(tokens[0]), num_threads=4, sample_rate=16000, feature_dim=80, provider=provider, **options)

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
