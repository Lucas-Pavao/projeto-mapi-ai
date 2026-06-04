import argparse
import uvicorn
from mapi_ai.trainer import train_pipeline

def main():
    parser = argparse.ArgumentParser(description="MAPI AI - Flood Prediction System")
    parser.add_argument("--mode", choices=["train", "serve"], default="serve", help="Mode: train or serve")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("Starting Training Pipeline...")
        train_pipeline()
    else:
        print("Starting Inference API...")
        uvicorn.run("mapi_ai.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
