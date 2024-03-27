from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def s1_to_rgb(vv_image, vh_image):
    ratio_image = np.clip(np.nan_to_num(vh_image / vv_image, 0), 0, 1)
    rgb_image = np.stack((vv_image, vh_image, 1 - ratio_image), axis=2)
    return rgb_image


def create_path_df(data_folder: Path, add_flood_label_paths=False):
    """Create dataframe with paths such that order of images is guaranteed to match for different applications
    Flood label paths can be added optionally because they are not always available.
    """
    filenames = [path.name.replace("_vh.png", "") for path in data_folder.glob("vh/*")]
    vh_paths = [data_folder / "vh" / f"{filename}_vh.png" for filename in filenames]
    vv_paths = [data_folder / "vv" / f"{filename}_vv.png" for filename in filenames]
    pred_paths = [data_folder / "output" / f"{filename}_pred.png" for filename in filenames]

    df_paths = pd.DataFrame({"vv_image_path": vv_paths, "vh_image_path": vh_paths, "pred_path": pred_paths})

    if add_flood_label_paths:
        flood_label_paths = [data_folder / "flood_labels" / f"{filename}.png" for filename in filenames]
        df_paths["flood_label_path"] = flood_label_paths

    return df_paths


class ETCIDataset(Dataset):
    """Dataset to be used with torch Dataloader. Translates the vv and vh projections to rgb images
    by overwriting Dataset.__getitem__()
    """

    def __init__(self, df_paths):
        self.dataset = df_paths

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        example = {}

        df_row = self.dataset.iloc[index]

        # load vv and vh images
        vv_image = cv2.imread(str(df_row["vv_image_path"]), 0) / 255.0
        vh_image = cv2.imread(str(df_row["vh_image_path"]), 0) / 255.0

        # convert vv and ch images to rgb
        rgb_image = s1_to_rgb(vv_image, vh_image)

        example["image"] = rgb_image.transpose((2, 0, 1)).astype("float32")

        return example
