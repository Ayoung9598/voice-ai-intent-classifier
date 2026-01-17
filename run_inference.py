#!/usr/bin/env python
"""
Inference Script for Voice AI Intent Classifier

Run this script to test the model interactively:
    python run_inference.py --model outputs/models/best_model.pt

Or start the API server:
    python run_inference.py --serve --port 8000
"""

import argparse


def interactive_mode(model_path: str):
    """Run interactive inference mode."""
    from src.models.intent_classifier import IntentClassifier
    from src.inference.predictor import IntentPredictor
    from src.inference.confidence import ConfidenceHandler
    
    print("\n" + "="*60)
    print("Voice AI Intent Classifier - Interactive Mode")
    print("="*60)
    print("Enter utterances to classify. Type 'quit' to exit.\n")
    
    # Load model
    print("Loading model...")
    predictor = IntentPredictor(model_path=model_path)
    confidence_handler = ConfidenceHandler()
    print("Model loaded!\n")
    
    while True:
        try:
            text = input("You: ").strip()
            
            if text.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            # Get prediction
            result = predictor.predict(text)
            processed = confidence_handler.process_prediction(result)
            
            print(f"\n  Intent: {processed['intent']}")
            print(f"  Confidence: {processed['confidence']:.2%}")
            print(f"  Action: {processed['action']}")
            print(f"  -> {processed['explanation']}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def demo_mode():
    """Run demo with sample utterances (no trained model required)."""
    from src.models.intent_classifier import create_model
    from src.inference.predictor import IntentPredictor
    from src.inference.confidence import ConfidenceHandler
    
    print("\n" + "="*60)
    print("Voice AI Intent Classifier - Demo Mode")
    print("="*60)
    print("(Using untrained model - predictions will be random)\n")
    
    # Create untrained model for demo
    model = create_model()
    predictor = IntentPredictor(model=model)
    confidence_handler = ConfidenceHandler()
    
    # Sample utterances
    samples = [
        "Ndashaka kureba status ya application yanjye.",
        "What are the requirements for passport?",
        "Help me with payment",
        "I want to speak to an agent",
        "Nashaka gufata appointment kuri driving license i Kigali.",
    ]
    
    for text in samples:
        result = predictor.predict(text)
        processed = confidence_handler.process_prediction(result)
        
        print(f"Input: {text}")
        print(f"  -> Intent: {processed['intent']}")
        print(f"  -> Confidence: {processed['confidence']:.2%}")
        print(f"  -> Action: {processed['action']}\n")
    
    print("Demo complete!")


def serve_api(model_path: str, host: str, port: int):
    """Start the FastAPI server."""
    import uvicorn
    from src.api.app import create_app, init_with_model
    from src.inference.predictor import IntentPredictor
    from src.inference.confidence import ConfidenceHandler
    
    print("\n" + "="*60)
    print("Voice AI Intent Classifier - API Server")
    print("="*60)
    
    if model_path:
        print(f"Loading model from: {model_path}")
        predictor = IntentPredictor(model_path=model_path)
        init_with_model(predictor)
    else:
        print("Warning: No model specified. API will return 503 errors.")
    
    app = create_app()
    
    print(f"\nStarting server at http://{host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs")
    print("Press Ctrl+C to stop.\n")
    
    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Voice AI Intent Classifier Inference")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start API server instead of interactive mode"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode with sample utterances"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for API server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server"
    )
    args = parser.parse_args()
    
    if args.demo:
        demo_mode()
    elif args.serve:
        serve_api(args.model, args.host, args.port)
    else:
        if not args.model:
            print("Error: --model path required for interactive mode")
            print("Use --demo for demo mode without a trained model")
            return
        interactive_mode(args.model)


if __name__ == "__main__":
    main()
