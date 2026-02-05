# src/main.py
"""
Entry point for the vehicle-driver matching pipeline.
Run: python -m src.main
"""
from core.pipeline import VehicleDriverPipeline
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", default="data/entry_frames", help="Directory with entry frames")
    parser.add_argument("--exit", default="data/exit_frames", help="Directory with exit frames")
    parser.add_argument("--output", default="data/outputs", help="Output directory")
    args = parser.parse_args()

    pipeline = VehicleDriverPipeline(
        entry_frames_path=args.entry,
        exit_frames_path=args.exit,
        output_path=args.output,
        verbose=True
    )

    res = pipeline.run_analysis()
    logging.info("Done. Summary:\n%s", res["stats"])

if __name__ == "__main__":
    main()