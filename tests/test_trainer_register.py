from clara_prototype.train import Trainer

import scripts.register_model as reg_mod


def test_trainer_register_invokes_register(monkeypatch, tmp_path):
    called = {}

    def fake_register(entry):
        called['entry'] = entry

    # Monkeypatch the register function
    monkeypatch.setattr(reg_mod, 'register', fake_register)

    t = Trainer(base_model='tiny', data='data/finetune/sample.jsonl', out_dir=str(tmp_path / 'out'))
    meta = t.train()
    t.register(meta)

    assert 'entry' in called
    assert called['entry']['base'] == 'tiny'
    assert 'path' in called['entry']
    assert called['entry']['run_id'] == meta['run_id']
