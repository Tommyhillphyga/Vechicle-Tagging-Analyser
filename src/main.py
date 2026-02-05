from core.pipeline import VehicleDriverPipeline

def main():
    pipeline = VehicleDriverPipeline(
        entry_source="data/entry_frames",
        exit_source="data/exit_frames"
    )
    pipeline.run()

if __name__ == "__main__":
    main()