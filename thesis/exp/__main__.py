import importlib
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m thesis.exp <sub-command> [args]")
        sys.exit(1)

    sub = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]          # strip sub-command
    try:
        mod = importlib.import_module(f"thesis.exp.{sub}")
    except ModuleNotFoundError:
        print(f"Unknown sub-command '{sub}'")
        sys.exit(1)

    mod.main()                                       # every sub-module exposes main()

if __name__ == "__main__":
    main()