import os
import tempfile
from artbench10 import ArtBench10

import torchvision
from tqdm.auto import tqdm

label_map = {
    'impressionism': 0,
    'realism': 1,
    'romanticism': 2,
    'expressionism': 3,
    'baroque': 4,
    'post_impressionism': 5,
    'art_nouveau': 6,
    'surrealism': 7,
    'ukiyo_e': 8,
    'renaissance': 9,
}

CLASSES = {v:k for k,v in label_map.items()}


def main():
    for split in ["train", "test"]:
        out_dir = f"cifar_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")
        dataset = ArtBench10(
            root="/home/alpaca/guided-diffusion/temp", train=split == "train", download=False
        )

        print("dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)


if __name__ == "__main__":
    main()
