import scripts.peft_train as peft_train


def test_parser_defaults():
    parser = peft_train.build_parser()
    args = parser.parse_args(["--base", "tiny", "--data", "tests/fixtures/tiny_data.jsonl"])
    assert args.bnb_bit == 8
    assert args.mode == "lora"
    assert not args.push_to_gcs
    assert not args.push_image_to_ar
