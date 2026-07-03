"""
Voice customization module for Sanctuary's speech synthesis
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging
from dataclasses import dataclass
from transformers import SpeechT5HifiGan, SpeechT5Processor, pipeline

logger = logging.getLogger(__name__)

@dataclass
class VoiceProfile:
    """Voice profile containing speaker embeddings and metadata"""
    name: str
    embeddings: torch.Tensor
    speaker_id: str
    characteristics: Dict[str, float]  # Voice characteristics (pitch, speed, etc)
    emotional_style: Dict[str, torch.Tensor]  # Emotion-specific embeddings

class VoiceCustomizer:
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize voice customization system"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if cache_dir is None:
            self.cache_dir = Path.home() / ".sanctuary" / "voices"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize vocoder
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        
        # Voice profiles
        self.voice_profiles = {}
        
    def extract_speaker_embeddings(self, audio_path: str, speaker_name: str) -> VoiceProfile:
        """
        Extract speaker embeddings from audio sample
        
        Args:
            audio_path: Path to voice sample audio file
            speaker_name: Name to identify this voice
            
        Returns:
            VoiceProfile with extracted embeddings
        """
        try:
            # Load and process audio
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            
            # Extract speaker embeddings
            with torch.no_grad():
                inputs = self.processor(
                    waveform.squeeze().numpy(),
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                embeddings = self.vocoder.encode_speaker(inputs.input_values)[0]
            
            # Create voice profile
            profile = VoiceProfile(
                name=speaker_name,
                embeddings=embeddings,
                speaker_id=speaker_name.lower().replace(" ", "_"),
                characteristics={
                    "pitch": 1.0,
                    "speed": 1.0,
                    "energy": 1.0
                },
                emotional_style={}
            )
            
            # Save profile
            self.voice_profiles[profile.speaker_id] = profile
            self._save_profile(profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error extracting speaker embeddings: {e}")
            raise
            
    def create_emotional_voice(
        self, 
        profile: VoiceProfile,
        emotion: str,
        intensity: float = 1.0
    ) -> VoiceProfile:
        """
        Create emotion-specific voice embeddings
        
        Args:
            profile: Base voice profile
            emotion: Target emotion
            intensity: Emotion intensity (0-1)
            
        Returns:
            Updated voice profile with emotional style
        """
        try:
            base_embeddings = profile.embeddings
            
            # Apply emotion-specific modifications
            if emotion == "happy":
                # Increase pitch and energy
                modified = base_embeddings * torch.tensor([1.1, 1.0, 1.2])
            elif emotion == "sad":
                # Lower pitch and energy
                modified = base_embeddings * torch.tensor([0.9, 1.0, 0.8])
            elif emotion == "angry":
                # Increase energy, slight pitch raise
                modified = base_embeddings * torch.tensor([1.05, 1.0, 1.4])
            else:
                modified = base_embeddings
                
            # Apply intensity
            modified = base_embeddings + (modified - base_embeddings) * intensity
            
            # Update profile
            profile.emotional_style[emotion] = modified
            self._save_profile(profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error creating emotional voice: {e}")
            raise
            
    def _save_profile(self, profile: VoiceProfile):
        """Save voice profile to cache"""
        profile_path = self.cache_dir / f"{profile.speaker_id}.pt"
        torch.save(profile.__dict__, profile_path)
        
    def load_profile(self, speaker_id: str) -> Optional[VoiceProfile]:
        """Load voice profile from cache"""
        profile_path = self.cache_dir / f"{speaker_id}.pt"
        if profile_path.exists():
            # Profile is saved as a plain dict of str/tensor/dict-of-float/dict-of-tensor
            # (VoiceProfile.__dict__), so weights_only=True is safe and blocks pickle RCE.
            data = torch.load(profile_path, weights_only=True)
            profile = VoiceProfile(**data)
            self.voice_profiles[speaker_id] = profile
            return profile
        return None
        
    def adjust_voice_characteristics(
        self,
        profile: VoiceProfile,
        pitch: Optional[float] = None,
        speed: Optional[float] = None,
        energy: Optional[float] = None
    ) -> VoiceProfile:
        """
        Adjust voice characteristics
        
        Args:
            profile: Voice profile to modify
            pitch: Pitch adjustment factor
            speed: Speed adjustment factor
            energy: Energy/volume adjustment factor
            
        Returns:
            Updated voice profile
        """
        if pitch is not None:
            profile.characteristics["pitch"] = max(0.5, min(2.0, pitch))
        if speed is not None:
            profile.characteristics["speed"] = max(0.5, min(2.0, speed))
        if energy is not None:
            profile.characteristics["energy"] = max(0.5, min(2.0, energy))
            
        self._save_profile(profile)
        return profile