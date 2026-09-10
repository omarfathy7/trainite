from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


class DatapointModel(BaseModel):
    """Tokenized UltraChat 200k sample used by Trainite's causal LM pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    target: str
    train_input_ids: torch.Tensor
    train_label_ids: torch.Tensor
    attention_mask: torch.Tensor
    eval_input_ids: torch.Tensor


class UltraChat200kTransform:
    """Convert an UltraChat 200k conversation into a causal LM datapoint."""

    def __init__(self, tokenizer: Any, max_length: int = 128, ignore_index: int = -100) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index

    @staticmethod
    def _render_messages(messages: list[dict[str, object]]) -> str:
        return "".join(
            f"{str(message.get('role', 'user')).capitalize()}: "
            f"{str(message.get('content', '')).strip()}\n"
            for message in messages
        )

    def __call__(self, sample: dict[str, object]) -> DatapointModel:
        messages = sample["messages"]
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise ValueError("messages must be a list of dictionaries")

        messages = [m for m in messages if str(m.get("content", "")).strip()]
        if not messages:
            raise ValueError("messages must contain at least one non-empty message")

        if str(messages[-1].get("role", "")).lower() != "assistant":
            raise ValueError("UltraChat SFT samples must end with an assistant message")

        target = str(messages[-1].get("content", "")).strip()
        prompt = self._render_messages(messages[:-1]) + "Assistant: "
        full_text = prompt + target

        full = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_tokens = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )

        token_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        prompt_ids = prompt_tokens["input_ids"]

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)
        train_attention_mask = torch.tensor(attention_mask[:-1], dtype=torch.long)

        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = self.ignore_index

        eval_ids = torch.tensor(prompt_ids, dtype=torch.long)

        return DatapointModel(
            source=prompt,
            target=target,
            train_input_ids=input_ids,
            train_label_ids=labels,
            attention_mask=train_attention_mask,
            eval_input_ids=eval_ids,
        )
