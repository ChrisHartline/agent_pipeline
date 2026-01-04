def test_trainer_importable():
    try:
        from clara_prototype.train import Trainer
    except Exception as e:
        raise AssertionError("Trainer import failed: " + str(e))

    t = Trainer(base_model="test", data="test")
    meta = t.train()
    assert "run_id" in meta
