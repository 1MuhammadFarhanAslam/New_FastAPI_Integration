from typing import List, Optional
import bittensor as bt
from pydantic import Field

class MusicGeneration(bt.Synapse):
    """
    A class that transforms textual descriptions into music using machine learning models such as 'facebook/musicgen-medium' and 'facebook/musicgen-large'. Extends bt.Synapse, facilitating its integration into a broader neural-based generative system.
    """
    class Config:
        """
        Configuration class that mandates validation on attribute assignments, ensuring correctness and reliability of data for MusicGeneration instances.
        """
        validate_assignment = True

    text_input: str = Field(
        default=None,
        title="Text Input",
        description="Textual directives or descriptions intended to guide the music generation process."
    )
    model_name: Optional[str] = Field(
        default=None,
        title="Model Name",
        description="The machine learning model employed for music generation. Supported models: 'facebook/musicgen-medium', 'facebook/musicgen-large'."
    )
    music_output: List = Field(
        default=None,
        title="Music Output",
        description="The resultant music data, encoded as a list, generated from the text input."
    )
    duration: int = Field(
        default=None,
        title="Duration",
        description="The length of the generated music piece, specified in seconds."
    )

    def deserialize(self) -> List:
        """
        Processes and returns the music_output into a format ready for audio rendering or further analysis.
        """
        return self
