"""Resolve the configured ASR without loading its weights."""
import os


def model_spec(name):
    if name == 'gipformer':
        from .gipformer import GipFormer
        return GipFormer, os.environ['GIPFORMER_MODEL_PATH']
    if name == 'zipformer':
        from .zipformer import ZipFormer
        return ZipFormer, os.environ['ZIPFORMER_MODEL_PATH']
    if name in ('phowhisper', 'phowhisper_large'):
        from .phowhisper import PhoWhisper
        return PhoWhisper, os.environ['PHOWHISPER_MODEL_PATH']
    raise ValueError(f'ASR không hỗ trợ: {name}')


def decoding_spec(name, environ=None):
    """Non-default decoding of a transducer from GIPFORMER_* / ZIPFORMER_* (empty dict = greedy, as before)."""
    from .gipformer import decoding_options
    if name in ('gipformer', 'zipformer'):
        return decoding_options(name.upper(), environ)
    environ = os.environ if environ is None else environ
    prefix = 'PHOWHISPER' if name.startswith('phowhisper') else name.upper()
    if any(environ.get(f'{prefix}_{key}') for key in ('DECODING', 'BEAM', 'HOTWORDS', 'HOTWORDS_SCORE')):
        raise ValueError(f'{name}: beam search / hotwords are only wired for the sherpa-onnx transducers')
    return {}
