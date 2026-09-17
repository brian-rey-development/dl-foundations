import numpy as np

from foundations.data import Dataset, Split, make_moons, standardize_split
from foundations.nn import RELU, SIGMOID, init_network
from foundations.training import TrainConfig, evaluate, iterate_batches, train


def test_batches_cover_dataset_exactly_once(rng: np.random.Generator) -> None:
    data = Dataset(x=np.arange(10).reshape(-1, 1).astype(float), y=np.arange(10).reshape(-1, 1))
    batches = list(iterate_batches(data, batch_size=4, rng=rng))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(np.concatenate([b.y for b in batches]).ravel()) == list(range(10))


def test_training_reduces_loss(rng: np.random.Generator) -> None:
    data = make_moons(300, noise=0.1, rng=rng)
    parts, _ = standardize_split(Split(train=data, val=data, test=data))
    network = init_network((2, 16, 1), RELU, SIGMOID, rng)
    initial_loss, _ = evaluate(network, parts.train)
    config = TrainConfig(epochs=30, batch_size=32, learning_rate=0.1)
    trained, history = train(network, parts, config, rng)
    final_loss, final_acc = evaluate(trained, parts.train)
    assert len(history) == 30
    assert final_loss < initial_loss * 0.5
    assert final_acc > 0.9
