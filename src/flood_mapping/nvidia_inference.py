from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import ttach as tta
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from flood_mapping.utils import ETCIDataset, create_path_df


def predict_with_single_model(model_def, weights, device, data_loader):
    """Makes predictions based on a single unet model"""
    model_def.load_state_dict(torch.load(weights, map_location=device))
    model = tta.SegmentationTTAWrapper(
        model_def, tta.aliases.d4_transform(), merge_mode="mean"
    )  # mean yields the best results
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    final_predictions = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # load image and mask into device memory
            image = batch["image"].to(device)
            # pass images into model
            pred = model(image)
            # add to final predictions
            final_predictions.append(pred.detach().cpu().numpy())

    final_predictions = np.concatenate(final_predictions, axis=0)

    return final_predictions


def predict_floods(
    data_folder: Path = Path("data/inference_sample"),
    batch_size: int = 1,
    num_workers: int = 0,
    save_output: bool = True,
):
    """Predict floods based on vv and vh polarization pngs

    Args:
        data_folder (Path, optional): Path which contains vv and vh folders with input pngs.
        The images in the vv and vh folders are expected to be named <img_name>_vv.png and <img_name>_vh.png.
        Defaults to Path("data/inference_sample").
        batch_size (int, optional): sets the batch size for the torch Dataloader. Defaults to 1.
        num_workers (int, optional): sets the num_workers for the torch Dataloader. Defaults to 0 meaning cpu.
        save_output (bool, optional): if True creates an output folder in your data_folder saving images and numpy forecasts
    """

    if not (data_folder / "vv").exists() or not (data_folder / "vh").exists():
        raise Exception("Make sure your data folder has subfolders vv and vh which contain the input images")

    if (
        not Path("models/unet_mobilenet_v2_0.pth").exists()
        or not Path("models/upp_mobilenetv2_0.pth").exists()
    ):
        raise Exception(
            """
            Make sure to have the base model weights saved to models/unet_mobilenet_v2_0.pth and
            models/upp_mobilenetv2_0.pth. The weights can be downloaded here:
            https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection/releases/download/v1.0.0/pretrained_weights.tar.gz
            """
        )

    df_paths = create_path_df(data_folder)

    # Create dataset with RGB images from the vv and vh projection
    etci_dataset = ETCIDataset(df_paths)

    data_loader = DataLoader(
        etci_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Set torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create list of unet models and their weights.
    # Always the 2 base models and possibly a pseudolabel trained one
    unet_mobilenet = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=2)
    upp_mobilenet = smp.UnetPlusPlus(
        encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=2
    )

    model_defs = [
        unet_mobilenet,
        upp_mobilenet,
    ]

    weight_paths = [
        Path("models/unet_mobilenet_v2_0.pth"),
        Path("models/upp_mobilenetv2_0.pth"),
    ]

    if (Path("models") / "unet_pseudolabel_model.pth").exists():
        print("Running with the base models and a pseudolabel trained model")
        unet_pseudo = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=2)
        model_defs.append(unet_pseudo)
        weight_paths.append(Path("models") / "unet_pseudolabel_model.pth")

    # Make predictions
    all_preds = []
    for defi, weights in zip(model_defs, weight_paths):
        all_preds.append(predict_with_single_model(defi, weights, device, data_loader))

    all_preds = np.array(all_preds)
    all_preds = np.mean(all_preds, axis=0)
    class_preds = all_preds.argmax(axis=1).astype("uint8")

    if save_output:
        # Save output images and numpy array of predictions
        (data_folder / "output").mkdir(parents=True, exist_ok=True)
        save_path = data_folder / "output" / "all_predictions.npy"
        np.save(save_path, class_preds, fix_imports=True, allow_pickle=False)

        for filename, pred in zip(df_paths["pred_path"], class_preds):
            plt.imsave(filename, pred)

    return class_preds


if __name__ == "__main__":
    # Original nvidia settings for gpu
    # batch_size = 96 * torch.cuda.device_count()
    # num_workers=os.cpu_count()

    batch_size = 1
    num_workers = 0
    predict_floods(data_folder=Path("data/inference_sample"), num_workers=num_workers, batch_size=batch_size)
