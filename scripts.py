from subprocess import run
import sys


def test():
    run(["flake8"], check=True)
    run(
        [
            "pytest",
            "--cov=pylinalg",
            "--cov-report=term-missing",
            "--cov-fail-under=95",
        ],
        check=True,
    )


def main():
    cmd = sys.argv[0]
    if cmd == "test":
        test()
    else:
        raise ValueError()


if __name__ == "__main__":
    main()
