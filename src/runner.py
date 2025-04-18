import argparse, time
from src.agent import run_once

parser = argparse.ArgumentParser(description="Run R0 trading loop.")
parser.add_argument("--loop", type=int, help="seconds between iterations")
args = parser.parse_args()

if args.loop:
    while True:
        print(run_once())
        time.sleep(args.loop)
else:
    print(run_once())
