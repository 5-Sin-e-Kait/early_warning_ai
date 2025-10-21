import sys
import pandas as pd
from pipeline import run_pipeline


def main():
    if len(sys.argv) < 4:
        print("Usage: python -m risk_model ds1.csv ds2.csv ds3.csv [preds.csv]")
        return
    ds1 = pd.read_csv(sys.argv[1])
    ds2 = pd.read_csv(sys.argv[2])
    ds3 = pd.read_csv(sys.argv[3])
    preds = pd.read_csv(sys.argv[4]) if len(sys.argv) >= 5 else None

    out = run_pipeline(ds1, ds2, ds3, preds)
    out.to_csv("risk_output.csv", index=False)
    print("Saved: risk_output.csv")


if __name__ == "__main__":
    main()