"""Run periodically with a scheduler. Retrain when enough labeled feedback arrives."""
import os
from pathlib import Path
import pandas as pd
from src.data import load_reference
from src.train import train

def main():
    incoming = Path("data/incoming/feedback.csv")
    if not incoming.exists():
        print("No feedback yet")
        return
    new = pd.read_csv(incoming)
    if len(new) < int(os.getenv("MIN_NEW_ROWS", "50")):
        print("Waiting for more labeled rows")
        return
    combined = pd.concat([load_reference(), new], ignore_index=True)
    train(combined)
    incoming.rename(incoming.with_name("processed_feedback.csv"))
    print("Retrained successfully")

if __name__ == "__main__":
    main()
