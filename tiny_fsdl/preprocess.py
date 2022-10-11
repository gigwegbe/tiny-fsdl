import logging
import os
from pathlib import Path
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

LOG = logging.getLogger(__name__)

###############################
#### Set up the parameters ####
###############################

# Image directory
INPUT_DIR = "./ml/data/raw_images/data1"
OUTPUT_DIR = "./ml/data/processed_images/data1"

# Relative locations
y = 11
w = 15
h = 25
digits_before = 4
digits_after = 2
x_before = 46
x_after = 89


#######################
#### Preprocessing ####
#######################


class Preprocess:
    def __init__(
        self,
        y=y,
        w=w,
        h=h,
        digits_before=digits_before,
        digits_after=digits_after,
        x_before=x_before,
        x_after=x_after,
    ):
        self.y = y
        self.w = w
        self.h = h
        self.digits_before = digits_before
        self.digits_after = digits_after
        self.x_before = x_before
        self.x_after = x_after

    def crop(self, img, x):
        return img[
            self.y : self.y + self.h,
            x : x + self.w,
        ]

    def process_image(self, img_path, output_dir):
        LOG.info(f"Cropping image {img_path}")
        img_path = Path(img_path)
        output_dir = Path(output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        assert os.path.exists(output_dir), "Output directory does not exist"

        raw_img = cv2.imread(str(img_path))

        # Digits before decimal
        for digit in range(self.digits_before):
            if digit < 0 or digit == 3 or digit == 4:
                continue
            x = x_before + (digit * 15)
            new_crop = self.crop(raw_img, x)
            out_file = Path(output_dir, f"cropped_{img_path.name}_b_{digit}.bmp")
            cv2.imwrite(
                filename=str(out_file),
                img=new_crop,
            )
            LOG.info(f"Wrote crop to {out_file}")

        # Digits after decimal
        for digit in range(self.digits_after):
            if digit < 0:
                continue
            x = self.x_after + (digit * 15)
            new_crop = self.crop(raw_img, x)
            out_file = Path(output_dir, f"cropped_{img_path.name}_a_{digit}.bmp")
            cv2.imwrite(
                filename=str(out_file),
                img=new_crop,
            )
            LOG.info(f"Wrote crop to {out_file}")

    def process_dir(
        self,
        input_dir,
        output_dir,
        digits_before,
        digits_after,
        x_before,
        x_after,
    ):
        LOG.info("Beginning preprocessing. . .")
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        assert os.path.exists(input_dir), "Input directory does not exist"
        assert os.path.exists(output_dir), "Output directory does not exist"

        crop_ix = 0
        for img_ix, fp in enumerate(tqdm(os.listdir(input_dir))):
            LOG.info(fp)
            raw_img = cv2.imread(str(Path(input_dir, fp)))
            # Digits before decimal
            for digit in range(digits_before):
                if digit < 0 or digit == 3 or digit == 4:
                    continue
                x = x_before + (digit * 15)
                new_img = self.crop(raw_img, x)
                cv2.imwrite(
                    filename=str(Path(OUTPUT_DIR, f"cropped_{fp}_b_{digit}.bmp")),
                    img=new_img,
                )
                crop_ix += 1

            # Digits after decimal
            for digit in range(digits_after):
                if digit >= 0:
                    x = x_after + (digit * 15)
                    new_img = self.crop(raw_img, x)
                    cv2.imwrite(
                        filename=str(Path(OUTPUT_DIR, f"cropped_{fp}_a_{digit}.bmp")),
                        img=new_img,
                    )
                    crop_ix += 1
        LOG.info(f"Finished preprocessing!")
        LOG.info(f"Total images {img_ix} and crops {crop_ix}")


##############
#### Main ####
##############


def main():
    preprocess = Preprocess(y, w, h)
    preprocess.start(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        digits_before=digits_before,
        digits_after=digits_after,
        x_before=x_before,
        x_after=x_after,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
