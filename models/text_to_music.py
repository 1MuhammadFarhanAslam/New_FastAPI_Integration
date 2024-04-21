# import torch
# from transformers import AutoProcessor, MusicgenForConditionalGeneration

# from audiocraft.models import MusicGen
# from audiocraft.data.audio import audio_write


# class MusicGenerator:
#     def __init__(self, model_path="facebook/musicgen-medium"):
#         self.model_name = model_path  
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.processor = AutoProcessor.from_pretrained(self.model_name)
#         # self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
#         self.model = MusicGen.get_pretrained(self.model_name)
#         # self.model.to(self.device)

#     def generate_music(self, prompt, duration):
#         self.model = self.model.set_generation_params(duration=duration)
#         text=[prompt],
#         try:
#             # inputs = self.processor(
#             #     padding=True,
#             #     return_tensors="pt",
#             # ).to(self.device)
#             audio_values = self.model.generate(text)
#             return audio_values[0, 0].cpu().numpy()
#         except Exception as e:
#             print(f"An error occurred with {self.model_name}: {e}")
#             return None


# import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class MusicGenerator:
    def __init__(self, model_path="facebook/musicgen-medium"):
        self.model_name = model_path  
        self.model = MusicGen.get_pretrained(self.model_name)
        # self.sample_rate = self.model.sample_rate

    def generate_music(self, prompt, duration=15): #, temperature=0.7, top_k=250
        """Generate music based on a given description."""
        self.model.set_generation_params(
            duration=duration,
            # temperature=temperature,
            # top_k=top_k
        )

        try:
            # Generate music from text description
            wav = self.model.generate([prompt])
            # Return the generated waveform for the first result
            wav = wav.cpu()
            return wav
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # def save_music(self, filename, audio):
    #     """Save generated audio to a file."""
    #     try:
    #         # Write the generated audio to a WAV file with loudness normalization
    #         audio_write(
    #             filename,
    #             audio,
    #             self.sample_rate,
    #             strategy="loudness"
    #         )
    #     except Exception as e:
    #         print(f"An error occurred while saving the file: {e}")
