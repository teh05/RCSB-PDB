from src.training.split import batched_slices, split_indices


def test_split_reproducible():
    a, b, t = split_indices(100, val_fraction=0.2, seed=0)
    c, d, u = split_indices(100, val_fraction=0.2, seed=0)
    assert list(a) == list(c)
    assert list(b) == list(d)
    assert list(t) == list(u)
    assert len(b) == 20
    assert len(a) == 80
    assert len(t) == 0
    assert set(a).isdisjoint(set(b))
    assert set(a).union(set(b)) == set(range(100))


def test_split_small_has_no_val():
    train, val, test = split_indices(3, val_fraction=0.2, seed=0)
    assert len(train) == 3
    assert len(val) == 0
    assert len(test) == 0


def test_batched_slices():
    assert batched_slices(10, 8) == [(0, 8), (8, 10)]


def test_split_val_fraction_zero_uses_all():
    train, val, test = split_indices(100, val_fraction=0.0, seed=0)
    assert len(train) == 100
    assert len(val) == 0
    assert len(test) == 0
    assert set(train.tolist()) == set(range(100))


def test_split_train_val_test():
    train, val, test = split_indices(100, val_fraction=0.1, test_fraction=0.1, seed=0)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
    assert set(train).isdisjoint(set(val))
    assert set(train).isdisjoint(set(test))
    assert set(val).isdisjoint(set(test))
    assert set(train).union(set(val)).union(set(test)) == set(range(100))
