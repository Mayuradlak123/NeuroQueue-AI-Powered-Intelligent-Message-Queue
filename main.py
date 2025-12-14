import uvicorn
import os

if __name__ == "__main__":
    # Ensure models exist
    if not os.path.exists("models/intent_model.pth"):
        print("⚠️  Models not found. Running initialization...")
        import neuroqueue.ai.train as trainer
        trainer.train_dummy_models()

    print("🚀 Starting NeuroQueue AI Node...")
    uvicorn.run("neuroqueue.api.app:app", host="0.0.0.0", port=8000, reload=True)
