


def test_float_inputs():
    # data is a floating number
    manager = TrainingManager(
        name="TestA", logdir="./EvalResults/",
        stopping_fn=lambda b, h: v not in h[-3:],
        updating_fn=lambda b, h, v: v > b,
        load_when_possible=False)

    # check initial values
    assert manager.empty
    assert manager.best_value is None
    assert manager.value_history is None
    assert manager.best_checkpoint is None
    try:
        manager.value_type
    except ValueError:
        pass

    try:
        manager.value_keys
    except ValueError:
        pass
        