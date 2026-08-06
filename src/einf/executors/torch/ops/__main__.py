import argparse

from einf.executors.torch.ops import custom_ops_available, load_custom_ops


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and load einf Torch custom ops")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_custom_ops(verbose=args.verbose)
    print(f"einf custom ops registered: {custom_ops_available()}")


if __name__ == "__main__":
    main()
