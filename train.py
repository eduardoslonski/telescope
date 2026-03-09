def main():
    import os
    from pathlib import Path

    from telescope.utils.config_loader import parse_args_and_load
    import telescope.utils.config as config_module

    config_module._cfg = parse_args_and_load()

    # Propagate checkpoint_dir to env before orchestrator import (paths.py reads it at import time)
    if config_module._cfg.checkpoint_dir is not None:
        os.environ["TELESCOPE_CHECKPOINT_DIR"] = str(
            Path(config_module._cfg.checkpoint_dir).resolve()
        )

    from telescope.orchestrator.orchestrator import main as orchestrator_main
    orchestrator_main()


if __name__ == "__main__":
    main()
