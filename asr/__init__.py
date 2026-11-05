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
