"""Standalone eval entry point — evaluate HF checkpoints after training.

Usage::

    uv run eval --config eval_run.yaml
    uv run eval --config eval_run.yaml --max_model_len 8000
"""


def main():
    from telescope.eval_standalone.config_loader import parse_args_and_load

    eval_cfg = parse_args_and_load()

    from telescope.eval_standalone.driver import run_eval

    run_eval(eval_cfg)


if __name__ == "__main__":
    main()
