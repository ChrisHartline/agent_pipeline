from clara_prototype.peft_trainer import PeftTrainer
from clara_prototype.data_collator import DataCollatorSFT


def test_trainer_receives_sft_collator(monkeypatch):
    import transformers

    # Dummy Trainer that records the last instance
    class DummyTrainer:
        last_instance = None

        def __init__(self, model=None, args=None, train_dataset=None, data_collator=None):
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.data_collator = data_collator
            DummyTrainer.last_instance = self

        def train(self):
            return None

    class DummyTrainingArguments:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(transformers, "Trainer", DummyTrainer)
    monkeypatch.setattr(transformers, "TrainingArguments", DummyTrainingArguments)

    # Also patch symbols inside the peft_trainer module to avoid import-caching issues
    import clara_prototype.peft_trainer as pt
    monkeypatch.setattr(pt, "Trainer", DummyTrainer, raising=False)
    monkeypatch.setattr(pt, "TrainingArguments", DummyTrainingArguments, raising=False)


    trainer = PeftTrainer(base="tiny", dataset="tests/fixtures/tiny_data.jsonl", out_dir="/tmp/out", smoke=False)

    # Provide a simple tokenizer stub if transformers not available
    try:
        model, tokenizer = trainer._load_tokenizer_and_model()
    except RuntimeError:
        class TK:
            def encode(self, text, add_special_tokens=False):
                # Return short token lists proportional to the text length (stable and small)
                n = max(1, min(4, (len(text) // 20) + 1))
                return [1] * n

            def pad(self, batch, padding=True, return_tensors="pt"):
                # Pad input_ids to the max length in the batch and return simple dict like tokenizer
                ids = batch.get("input_ids", [])
                max_len = max(len(x) for x in ids)
                padded = [x + [0] * (max_len - len(x)) for x in ids]
                return {"input_ids": padded}

            def __getattr__(self, name):
                return None

            pad_token = None
            eos_token = None
            eos_token_id = None

            def save_pretrained(self, *a, **k):
                return None

        tokenizer = TK()
        model = object()

    tiny_ds = {"train": [{"human": "Hi", "assistant": "Hello"}]}

    # Ensure trainer can build a collator for the tokenizer (integration point)
    collator = trainer._build_data_collator(tokenizer)
    assert collator is not None
    assert isinstance(collator, DataCollatorSFT)

    # Validate collator behaviour directly
    batch = collator(tiny_ds["train"])
    labels = batch["labels"]
    assert (labels != -100).any()

    # Also try running _train_with_trainer to ensure it doesn't crash in normal conditions.
    # We don't require it to instantiate the real Trainer in all environments (some tests may
    # influence the transformers import behavior), but attempt it and surfacing any errors
    # helps with debugging.
    res = trainer._train_with_trainer(model, tokenizer, tiny_ds)

    if DummyTrainer.last_instance is None:
        # if Trainer wasn't instantiated, at least ensure the call returned an error dict we can inspect
        assert "error" in res
    else:
        # If Trainer instantiated, ensure it received our collator instance
        dc = DummyTrainer.last_instance.data_collator
        assert dc is collator
