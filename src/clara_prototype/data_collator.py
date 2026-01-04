from dataclasses import dataclass
from typing import List, Dict

import torch


@dataclass
class DataCollatorSFT:
    tokenizer: object
    max_length: int

    def _ensure_pad_token(self):
        if self.tokenizer.pad_token is None:
            # fallback to eos token if pad not set
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # As a last resort, set pad token to unk
                self.tokenizer.pad_token = self.tokenizer.unk_token or "<pad>"

    def __call__(self, features: List[Dict]):
        """Build batch with padded input_ids, attention_mask and labels.

        Accepts examples in one of the forms:
        - {'human': str, 'assistant': str}
        - {'text': 'Human: ...\nAssistant: ...'}
        """
        self._ensure_pad_token()

        inputs = []
        labels_list = []
        for ex in features:
            if "human" in ex and "assistant" in ex:
                human = ex["human"]
                assistant = ex["assistant"]
            elif "text" in ex:
                # split at the assistant marker
                parts = ex["text"].split("\nAssistant:")
                if len(parts) == 2:
                    human = parts[0].replace("Human:", "").strip()
                    assistant = parts[1].strip()
                else:
                    # fallback: whole text as assistant
                    human = ""
                    assistant = ex["text"]
            else:
                raise ValueError("Feature must contain 'human' and 'assistant' or 'text'")

            # Build using encode separately to ensure EOS placement
            prefix_ids = self.tokenizer.encode(f"Human: {human}\nAssistant: ", add_special_tokens=False)
            assistant_ids = self.tokenizer.encode(assistant, add_special_tokens=False)
            if self.tokenizer.eos_token_id is not None:
                assistant_ids = assistant_ids + [self.tokenizer.eos_token_id]

            # Truncate preserving assistant tokens
            if len(prefix_ids) + len(assistant_ids) > self.max_length:
                if len(assistant_ids) >= self.max_length:
                    assistant_ids = assistant_ids[-self.max_length:]
                    prefix_ids = []
                else:
                    allowed_prefix = self.max_length - len(assistant_ids)
                    prefix_ids = prefix_ids[-allowed_prefix:]

            input_ids = prefix_ids + assistant_ids
            labels = [-100] * len(prefix_ids) + assistant_ids.copy()

            inputs.append(input_ids)
            labels_list.append(labels)

        # Pad sequences
        batch = self.tokenizer.pad({"input_ids": inputs}, padding=True, return_tensors="pt")

        # Ensure input_ids are tensors (some test stubs return lists)
        ids = batch.get("input_ids")
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.long)
            batch["input_ids"] = ids

        # Ensure attention_mask exists
        if "attention_mask" not in batch:
            # If tokenizer exposes a pad token id, compute mask; otherwise default to ones
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                batch["attention_mask"] = (ids != pad_id).long()
            else:
                batch["attention_mask"] = torch.ones_like(ids, dtype=torch.long)

        # Build labels tensor with -100 padding
        labels_p = torch.full_like(batch["input_ids"], fill_value=-100)
        for i, lab in enumerate(labels_list):
            labels_p[i, : len(lab)] = torch.tensor(lab, dtype=labels_p.dtype)

        batch["labels"] = labels_p
        return batch
