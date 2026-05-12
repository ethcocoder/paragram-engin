import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def download_hd_test_images(num_images=10, target_dir="dataset/flickr8k"):
    """
    Downloads real-life HD images from Unsplash for realistic testing.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Curated list of high-detail / high-texture images
    # 1. Nature/Grass (Texture)
    # 2. Architecture (Lines/Math)
    # 3. Portrait (Detail)
    # 4. Fabric (Texture)
    # 5. Smooth Sky (Vacuum)
    
    queries = [
        "nature,grass", "architecture,lines", "portrait,face", 
        "texture,fabric", "landscape,sky", "city,night", 
        "forest,leaves", "ocean,waves", "mountain,snow", "macro,eye"
    ]
    
    print(f"🚀 Downloading {num_images} Real-Life HD images...")
    
    for i in tqdm(range(num_images)):
        query = queries[i % len(queries)]
        url = f"https://source.unsplash.com/featured/1024x1024?{query}"
        
        try:
            # Note: Unsplash Source API might be redirected or rate-limited. 
            # Using a more stable placeholder if needed, but let's try direct.
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                # Ensure it's RGB
                img = img.convert('RGB')
                img.save(os.path.join(target_dir, f"real_hd_{i}.png"))
            else:
                print(f"Failed to download image {i} (Status: {response.status_code})")
        except Exception as e:
            print(f"Error downloading image {i}: {e}")

if __name__ == "__main__":
    download_hd_test_images()
