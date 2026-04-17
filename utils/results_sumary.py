import json
import pandas as pd
from pathlib import Path


class ResultsDataLoader:
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> dict:
        """Load JSON results file."""
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} does not exist")
        
        with open(self.path, "r") as f:
            data = json.load(f)

        # basic validation
        if not isinstance(data, dict):
            raise ValueError("Expected top-level dict of test_id -> list[bool]")
        
        return data


def summarize_results(data: dict) -> pd.DataFrame:
    """Convert raw results into tabular summary."""
    rows = []

    for test_id, outcomes in data.items():
        if not isinstance(outcomes, list):
            raise ValueError(f"Test {test_id} is not a list")

        total = len(outcomes)
        successes = sum(outcomes)
        success_rate = successes / total if total > 0 else 0.0

        rows.append({
            "test_id": int(test_id),
            "num_trials": total,
            "num_success": successes,
            "success_rate": success_rate
        })

    df = pd.DataFrame(rows)
    return df.sort_values(by="test_id").reset_index(drop=True)


def main(path: str):
    loader = ResultsDataLoader(path)
    data = loader.load()

    df = summarize_results(data)

    print(df.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--reuslts_path")
    args = parser.parse_args()
    # replace with your actual file path
    main(args.reuslts_path)