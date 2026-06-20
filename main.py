import argparse
import uvicorn
from mapi_ai.trainer import train_pipeline

def main():
    parser = argparse.ArgumentParser(description="MAPI AI - Flood Prediction System")
    parser.add_argument("--mode", choices=["train", "serve"], default="serve", help="Mode: train or serve")
    parser.add_argument("--csv", type=str, help="Path to scenario labels CSV file")
    parser.add_argument("--model", type=str, choices=["all", "xgb", "lstm"], default="all",
                        help="Model to train: xgb, lstm, or all (default: all)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print(f"Starting Training Pipeline for model={args.model}...")
        train_pipeline(csv_path=args.csv, model_type=args.model)
    else:
        print("Starting Inference API...")
        uvicorn.run("mapi_ai.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
