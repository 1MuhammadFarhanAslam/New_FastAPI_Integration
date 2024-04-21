import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


class MusicGenerator:
    def __init__(self, model_path="facebook/musicgen-medium"):
        self.model_name = model_path  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.processor = AutoProcessor.from_pretrained(self.model_name)
        # self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
        self.model = MusicGen.get_pretrained(self.model_name)
        self.model.to(self.device)

    def generate_music(self, prompt, duration):
        self.model = self.model.set_generation_params(duration=duration)
        text=[prompt],
        try:
            # inputs = self.processor(
            #     padding=True,
            #     return_tensors="pt",
            # ).to(self.device)
            audio_values = self.model.generate(text)
            return audio_values[0, 0].cpu().numpy()
        except Exception as e:
            print(f"An error occurred with {self.model_name}: {e}")
            return None
