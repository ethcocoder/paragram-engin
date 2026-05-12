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
        # Using picsum.photos for high stability in automated tests
        url = f"https://picsum.photos/1024/1024?random={i}"
        
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                # Ensure it's RGB
                img = img.convert('RGB')
                img.save(os.path.join(target_dir, f"real_hd_{i}.png"))
            else:
                print(f"Failed to download image {i} (Status: {response.status_code})")
        except Exception as e:
            print(f"Error downloading image {i}: {e}")

def download_flickr8k(target_dir="dataset/flickr8k"):
    """
    Downloads the Flickr8k dataset (images only) from a public mirror.
    """
    os.makedirs(target_dir, exist_ok=True)
    images_dir = os.path.join(target_dir, "images")
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 8000:
        print("✅ Flickr8k already exists and looks complete.")
        return

    print("🚀 Downloading Flickr8k Dataset (~1.1GB)...")
    # Using a reliable public mirror for Flickr8k
    url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    
    zip_path = os.path.join(target_dir, "flickr8k.zip")
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, "wb") as f, tqdm(
        desc="Flickr8k.zip",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
            
    print("📦 Extracting Flickr8k...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    
    # Rename folder if necessary to match expectations
    extracted_dir = os.path.join(target_dir, "Flicker8k_Dataset")
    if os.path.exists(extracted_dir):
        os.rename(extracted_dir, images_dir)
        
    os.remove(zip_path)
    print(f"✨ Flickr8k ready at {images_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--flickr8k":
        download_flickr8k()
    else:
        download_hd_test_images()
