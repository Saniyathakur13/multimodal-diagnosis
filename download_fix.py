import sys
import time
import requests

def download_with_retry():
    """Download model with retry logic"""
    
    print("📥 Downloading required models...")
    print("This may take 5-10 minutes depending on your internet speed")
    
    # First, upgrade huggingface hub
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface-hub"])
    
    # Set longer timeout
    import os
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    
    from transformers import AutoTokenizer, AutoModel
    
    # Try with different mirror
    models_to_download = [
        "distilbert-base-uncased",  # Small, fast model
    ]
    
    for model_name in models_to_download:
        print(f"\n🔄 Downloading {model_name}...")
        try:
            # Download with retry
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                resume_download=True,
                use_auth_token=None
            )
            model = AutoModel.from_pretrained(
                model_name,
                resume_download=True,
                use_auth_token=None
            )
            print(f"✅ Successfully downloaded {model_name}")
        except Exception as e:
            print(f"⚠️ Error downloading {model_name}: {e}")
            print("Trying alternative method...")
            
            # Alternative: Use local cache or different source
            try:
                # Use HuggingFace mirror
                import requests
                from transformers import AutoTokenizer, AutoModel
                
                # Custom download with progress
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    mirror='https://hf-mirror.com'
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    mirror='https://hf-mirror.com'
                )
                print(f"✅ Successfully downloaded {model_name} from mirror")
            except:
                print(f"❌ Failed to download {model_name}")
                print("Please check your internet connection and try again")
                return False
    
    print("\n✅ All models downloaded successfully!")
    return True

if __name__ == "__main__":
    download_with_retry()