import argparse
from dataclasses import replace
from pathlib import Path

from foundations.config import DEFAULT_CONFIG, ExperimentConfig
from foundations.experiment import ExperimentResult, run_experiment, to_artifact
from foundations.nn import count_parameters
from foundations.persistence import save_artifact
from foundations.training import History, TrainConfig
from foundations.visualization import plot_decision_boundary, plot_history

CLI_DESCRIPTION = "Train a from-scratch neural network on the two moons dataset."
GRADIENT_CHECK_TOLERANCE = 1e-6
LOG_EVERY = 20
MODEL_FILENAME = "model.npz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.training.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.training.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.training.learning_rate)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    training = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    return replace(DEFAULT_CONFIG, seed=args.seed, training=training, output_dir=args.output_dir)


def _print_history(history: History) -> None:
    for epoch in range(LOG_EVERY - 1, len(history), LOG_EVERY):
        m = history[epoch]
        print(
            f"epoch {epoch + 1:4d} | train loss {m.train_loss:.4f} acc {m.train_accuracy:.3f}"
            f" | val loss {m.val_loss:.4f} acc {m.val_accuracy:.3f}"
        )


def _print_report(result: ExperimentResult, config: ExperimentConfig) -> None:
    status = "OK" if result.gradient_check_error < GRADIENT_CHECK_TOLERANCE else "FAILED"
    print(f"network {config.model.layer_sizes} | {count_parameters(result.network)} parameters")
    print(f"gradient check | relative error {result.gradient_check_error:.2e} ({status})\n")
    _print_history(result.history)
    cm = result.test.confusion
    print(f"\ntest loss {result.test.loss:.4f} | test accuracy {result.test.accuracy:.3f}")
    print(
        f"confusion | TP={cm.true_positive} TN={cm.true_negative}"
        f" FP={cm.false_positive} FN={cm.false_negative}"
    )
    print(f"precision {cm.precision:.3f} | recall {cm.recall:.3f}")


def _save_plots(result: ExperimentResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_history(result.history, output_dir / "training_curves.png")
    plot_decision_boundary(result.network, result.data.test, output_dir / "decision_boundary.png")
    print(f"\nplots saved to {output_dir.resolve()}")


def _save_model(result: ExperimentResult, config: ExperimentConfig) -> None:
    path = config.output_dir / MODEL_FILENAME
    save_artifact(to_artifact(result, config), path)
    print(f"model saved to {path.resolve()}")


def main() -> None:
    config = _build_config(_parse_args())
    result = run_experiment(config)
    _print_report(result, config)
    _save_plots(result, config.output_dir)
    _save_model(result, config)


if __name__ == "__main__":
    main()
