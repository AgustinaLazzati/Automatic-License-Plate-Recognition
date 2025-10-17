import os
import numpy as np
import cv2 # OpenCV for saving images
import argparse

def generate_random_images(output_dir, num_images, img_size=(512, 512), mean=128, std_dev=30):
    """
    Generates random RGB images filled with Gaussian noise and saves them
    to a specified directory.

    Args:
        output_dir (str): The directory where the images will be saved.
        num_images (int): The number of images to generate.
        img_size (tuple): A tuple (width, height) for the image dimensions.
        mean (int): The mean pixel value for the Gaussian noise (0-255).
        std_dev (int): The standard deviation for the Gaussian noise.
                       A higher value means more variation/noise.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")
    else:
        print(f"Output directory '{output_dir}' already exists.")

    print(f"Generating {num_images} random images of size {img_size[0]}x{img_size[1]}...")

    for i in range(num_images):
        # Generate Gaussian noise for an RGB image
        # Each channel (R, G, B) will have its own independent noise
        # The shape is (height, width, channels)
        
        # We generate float numbers first
        noise = np.random.normal(mean, std_dev, (img_size[1], img_size[0], 3))
        
        # Clip values to ensure they are within valid 0-255 range
        # and convert to uint8 (8-bit unsigned integer) which is standard for images
        random_image = np.clip(noise, 0, 255).astype(np.uint8)

        # Construct filename and save
        filename = f"random_image_{i+1:04d}.png" # e.g., random_image_0001.png
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, random_image)
        
        if (i + 1) % 10 == 0 or (i + 1) == num_images:
            print(f"  Generated {i+1}/{num_images} images...")

    print(f"\n✅ Finished generating {num_images} random images in '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a specified number of random RGB images with Gaussian noise."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the generated images will be saved."
    )
    parser.add_argument(
        "-n", "--num_images",
        type=int,
        default=50,
        help="Number of random images to generate (default: 50)."
    )
    parser.add_argument(
        "-s", "--size",
        type=int,
        default=512,
        help="Side length for square images (e.g., 512 for 512x512, default: 512)."
    )
    parser.add_argument(
        "--mean",
        type=int,
        default=128,
        help="Mean pixel value for Gaussian noise (0-255, default: 128)."
    )
    parser.add_argument(
        "--std_dev",
        type=int,
        default=30,
        help="Standard deviation for Gaussian noise (default: 30). Higher = more variation."
    )

    args = parser.parse_args()
    
    generate_random_images(
        args.output_dir,
        args.num_images,
        img_size=(args.size, args.size),
        mean=args.mean,
        std_dev=args.std_dev
    )