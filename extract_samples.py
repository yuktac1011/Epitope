import os
from backend.app.kyc.data import load_kyc_data

def extract_samples():
    print("Loading dataset to extract sample images...")
    dataset = load_kyc_data("validation", max_samples=500)
    
    saved_real = False
    saved_fake = False
    
    for item in dataset:
        img = item["image"]
        label = item["label"]
        
        # Label 1 is Real, 0 is Fake
        if label == 1 and not saved_real:
            img.save("test_real_photo.png")
            saved_real = True
            print("Saved 'test_real_photo.png'")
            
        elif label == 0 and not saved_fake:
            img.save("test_fake_photo.png")
            saved_fake = True
            print("Saved 'test_fake_photo.png'")
            
        if saved_real and saved_fake:
            break

if __name__ == "__main__":
    extract_samples()
