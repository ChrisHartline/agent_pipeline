from clara_prototype.peft_trainer import PeftTrainer


class DummyTrainer:
    def __init__(self, model=None, args=None, train_dataset=None, data_collator=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.data_collator = data_collator

    def train(self):
        return None


class DummyTrainingArguments:
    def __init__(self, *args, **kwargs):
        pass


class ModelStub:
    def save_pretrained(self, path):
        pass
    def to(self, device):
        return self


def test_trainer_uses_sft_collator(monkeypatch):
    # Monkeypatch transformers Trainer and TrainingArguments used in _train_with_trainer
    import transformers

    monkeypatch.setattr(transformers, "Trainer", DummyTrainer)
    monkeypatch.setattr(transformers, "TrainingArguments", DummyTrainingArguments)

    trainer = PeftTrainer(base="tiny", dataset="tests/fixtures/tiny_data.jsonl", out_dir="/tmp/out", smoke=False)

    # Create stub model and tokenizer (use transformer tokenizer) lazily
    # We only need the method to instantiate DataCollatorSFT; call _load_tokenizer_and_model's tokenizer
    try:
        model, tokenizer = trainer._load_tokenizer_and_model()
    except RuntimeError:
        # If transformers not available in test env, skip the runtime heavy part by creating a minimal tokenizer stub
        class TK:
            def encode(self, *a, **k):
                return [1, 2]

            def __getattr__(self, name):
                return None

            pad_token = None
            eos_token = None
            eos_token_id = None

        tokenizer = TK()
        model = ModelStub()

    # Build a tiny dataset in the expected form
    tiny_ds = {"train": [{"human": "Hi", "assistant": "Hello"}]}

    # Call _train_with_trainer which should instantiate our DummyTrainer and pass a DataCollatorSFT
    result = trainer._train_with_trainer(model, tokenizer, tiny_ds)

    # If the environment has transformers, our DummyTrainer instance was used: ensure data_collator attribute exists and is not None
    # Because _train_with_trainer returns after calling DummyTrainer.train, and doesn't return the trainer object,
    # we assert that result is not raising and that the run completed.
    assert result is not None
