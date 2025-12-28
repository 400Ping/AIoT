from tools import main as cli_main

if __name__ == "__main__":
    # Mirror old behavior: run the benchmark subcommand with defaults.
    cli_main(["benchmark"])
