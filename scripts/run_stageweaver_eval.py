from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("client.stageweaver_runner", run_name="__main__")


if __name__ == "__main__":
    main()
