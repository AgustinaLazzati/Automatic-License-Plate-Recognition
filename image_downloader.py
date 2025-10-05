import os
import requests
import argparse
from urllib.parse import urlparse

def download_images_from_file(url_file, output_dir):
    """
    Reads a text file of URLs and downloads each image into a specified directory.

    Args:
        url_file (str): The path to the .txt file containing image URLs.
        output_dir (str): The directory where the images will be saved.
    """
    # --- 1. Setup: Create the output directory if it doesn't exist ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Saving images to directory: '{output_dir}'")
    except OSError as e:
        print(f"❌ Error creating directory {output_dir}: {e}")
        return

    # --- 2. Read URLs from the input file ---
    try:
        with open(url_file, 'r') as f:
            # Read lines and filter out any empty ones
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Error: The file '{url_file}' was not found.")
        return
        
    if not urls:
        print("⚠️ Warning: The URL file is empty. No images to download.")
        return

    print(f"Found {len(urls)} URLs to process.")
    
    # --- 3. Loop through URLs and download images ---
    for i, url in enumerate(urls):
        print(f"\n[{i+1}/{len(urls)}] 📥 Processing URL: {url}")
        try:
            # Make the request to the URL with a timeout
            response = requests.get(url, stream=True, timeout=15)
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()

            # --- 4. Determine a valid filename ---
            # Parse the URL to get the path
            path = urlparse(url).path
            # Get the last part of the path as the filename
            filename = os.path.basename(path)
            
            # If the filename is empty (e.g., URL is a root path), create a fallback name
            if not filename:
                # Try to get extension from content type, default to .jpg
                content_type = response.headers.get('content-type')
                if content_type and 'image' in content_type:
                    ext = '.' + content_type.split('/')[-1]
                else:
                    ext = '.jpg'
                filename = f"image_{i+1:04d}{ext}"
                print(f"   - No filename in URL, using fallback: '{filename}'")

            filepath = os.path.join(output_dir, filename)

            # --- 5. Save the image content to a file ---
            with open(filepath, 'wb') as img_file:
                # Write the content in chunks for efficiency with large files
                for chunk in response.iter_content(chunk_size=8192):
                    img_file.write(chunk)
            
            print(f"   ✅ Success! Image saved as '{filename}'")

        except requests.exceptions.RequestException as e:
            print(f"   ❌ Failed to download. Error: {e}")

    print("\n🎉 Download process complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download images from a list of URLs in a text file."
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="Path to the .txt file containing image URLs (one URL per line)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the downloaded images will be saved."
    )

    args = parser.parse_args()
    download_images_from_file(args.file, args.output_dir)