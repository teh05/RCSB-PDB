from src.training.split import batched_slices, split_indices


def test_split_reproducible():
    a, b = split_indices(100, val_fraction=0.2, seed=0)
    c, d = split_indices(100, val_fraction=0.2, seed=0)
    assert list(a) == list(c)
    assert list(b) == list(d)
    assert len(b) == 20
    assert len(a) == 80
    assert set(a).isdisjoint(set(b))
    assert set(a).union(set(b)) == set(range(100))


def test_split_small_has_no_val():
    train, val = split_indices(3, val_fraction=0.2, seed=0)
    assert len(train) == 3
    assert len(val) == 0


def test_batched_slices():
    assert batched_slices(10, 8) == [(0, 8), (8, 10)]
