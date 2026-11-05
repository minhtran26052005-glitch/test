"""hynt ZipFormer FP32 ONNX transducer, with the same decoding as GipFormer."""
from pathlib import Path
from .gipformer import GipFormer


class ZipFormer(GipFormer):
    def __init__(self, model_path: str, device: str = 'cpu'):
        path = Path(model_path)
        if not path.is_dir() or not (path / 'tokens.txt').is_file():
            raise RuntimeError(
                f'ZipFormer cần ONNX encoder/decoder/joiner và tokens.txt trong {path}. '
                'Với checkpoint hynt, config.json thực chất là bảng token; '
                'kiểm tra rồi sao chép nguyên nội dung sang tokens.txt khi tải model.'
            )
        super().__init__(str(path), device)
