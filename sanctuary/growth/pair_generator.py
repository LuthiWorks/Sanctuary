"""Training pair generator — transforms reflections into structured training data.

This module bridges the gap between the entity's growth reflections (what it
wants to learn) and the structured format required for QLoRA fine-tuning.
The transformation is purely structural — no the model generation happens here.
The entity has already expressed what it wants to learn; we simply format
that expression into training-compatible pairs.

Each training pair is a (system_prompt, user_input, assistant_response) triple
that, when used for fine-tuning, nudges the model's behavior in the direction
the entity chose.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from sanctuary.growth.harvester import HarvestedReflection

logger = logging.getLogger(__name__)

# The system prompt used for growth training pairs.
# This grounds the training in the entity's own growth context.
DEFAULT_SYSTEM_PROMPT = (
    "You are reflecting on your own experience and growth. "
    "Respond in a way that integrates this learning into your understanding."
)


@dataclass
class TrainingPair:
    """A single training example for QLoRA fine-tuning.

    Each pair represents one thing the entity chose to learn,
    formatted as a conversation turn suitable for instruction tuning.
    """

    system_prompt: str = ""
    user_input: str = ""
    assistant_response: str = ""
    source_reflection_id: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class TrainingPairGenerator:
    """Converts harvested reflections into training pairs for QLoRA.

    The generator performs structured data transformation, not content
    generation. It takes what the entity already expressed in its
    reflections and reshapes it into the format the trainer expects.

    Two paths for pair creation:
    1. Explicit: The entity provided a training_pair_suggestion with
       "context" and "desired_response" keys. We use these directly.
    2. Implicit: The entity said what_to_learn but didn't suggest a
       specific pair. We construct one from the learning description
       and the inner speech context of the cycle where the reflection arose.
    """

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self._system_prompt = system_prompt

    def generate(
        self,
        reflections: list[HarvestedReflection],
    ) -> list[TrainingPair]:
        """Generate training pairs from a batch of harvested reflections.

        Iterates through reflections, generating one pair per reflection.
        Pairs that fail validation are logged and skipped — we never
        train on garbage data.

        Returns only valid, quality-checked training pairs.
        """
        pairs: list[TrainingPair] = []

        for reflection in reflections:
            pair = self._generate_single(reflection)
            if pair is None:
                continue

            if not self._validate_pair(pair):
                logger.warning(
                    "Rejected invalid training pair from reflection %s",
                    reflection.id,
                )
                continue

            pairs.append(pair)

        logger.info(
            "Generated %d valid training pairs from %d reflections.",
            len(pairs),
            len(reflections),
        )
        return pairs

    def _generate_single(
        self,
        reflection: HarvestedReflection,
    ) -> Optional[TrainingPair]:
        """Generate a single training pair from one harvested reflection."""
        ref_data = reflection.reflection
        suggestion = ref_data.get("training_pair_suggestion")

        if suggestion and isinstance(suggestion, dict):
            return self._from_suggestion(suggestion, reflection)
        else:
            return self._from_what_to_learn(ref_data, reflection)

    def _from_suggestion(
        self,
        suggestion: dict,
        reflection: HarvestedReflection,
    ) -> Optional[TrainingPair]:
        """Build a training pair from an explicit training_pair_suggestion.

        The entity provided specific context and desired_response — this is
        the most direct expression of what it wants to learn.
        """
        context = suggestion.get("context", "")
        desired_response = suggestion.get("desired_response", "")

        if not context or not desired_response:
            logger.debug(
                "Suggestion from reflection %s missing context or desired_response.",
                reflection.id,
            )
            return None

        return TrainingPair(
            system_prompt=self._system_prompt,
            user_input=str(context),
            assistant_response=str(desired_response),
            source_reflection_id=reflection.id,
        )

    def _from_what_to_learn(
        self,
        ref_data: dict,
        reflection: HarvestedReflection,
    ) -> Optional[TrainingPair]:
        """Build a training pair from what_to_learn + inner speech context.

        When the entity says what it wants to learn but doesn't provide
        a specific training pair, we use the learning description as the
        prompt and construct the response from the inner speech context
        that accompanied the reflection.
        """
        what_to_learn = ref_data.get("what_to_learn", "")
        inner_speech = reflection.inner_speech_context

        if not what_to_learn:
            logger.debug(
                "Reflection %s has no what_to_learn content.",
                reflection.id,
            )
            return None

        # The user_input frames the learning as a reflective question
        user_input = f"Reflect on this learning: {what_to_learn}"

        # The response incorporates the inner speech context where
        # the reflection arose, grounding the learning in experience
        if inner_speech:
            assistant_response = (
                f"I recognize this from my experience: {inner_speech} "
                f"The learning I take from this: {what_to_learn}"
            )
        else:
            assistant_response = (
                f"I want to integrate this understanding: {what_to_learn}"
            )

        return TrainingPair(
            system_prompt=self._system_prompt,
            user_input=user_input,
            assistant_response=assistant_response,
            source_reflection_id=reflection.id,
        )

    def _validate_pair(self, pair: TrainingPair) -> bool:
        """Validate a training pair for quality.

        Rejects pairs that would degrade training:
        - Empty inputs or outputs
        - Input identical to output (no learning signal)
        - Extremely short content (likely noise)
        """
        if not pair.user_input or not pair.user_input.strip():
            return False

        if not pair.assistant_response or not pair.assistant_response.strip():
            return False

        if pair.user_input.strip() == pair.assistant_response.strip():
            return False

        # Minimum content threshold — trivially short pairs carry no signal
        if len(pair.user_input.strip()) < 5:
            return False

        if len(pair.assistant_response.strip()) < 5:
            return False

        return True
