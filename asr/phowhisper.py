"""PhoWhisper inference."""
class PhoWhisper:
    def __init__(self, model_path: str, device: str = 'cpu'):
        try:
            import torch
            from transformers import pipeline
            self.model = pipeline('automatic-speech-recognition', model=model_path, device=device, torch_dtype=torch.float32)
        except Exception as exc:
            raise RuntimeError(f'Không load được PhoWhisper {model_path}: {exc}') from exc

    def transcribe(self, audio_path: str) -> str:
        return str(self.model(audio_path, generate_kwargs={'language': 'vi', 'task': 'transcribe'})['text']).strip()
