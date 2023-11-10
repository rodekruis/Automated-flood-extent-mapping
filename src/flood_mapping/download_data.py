import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)

load_dotenv()

config = SHConfig()
config.sh_client_id = os.environ["SENTINELHUB_CLIENT_ID"]
config.sh_client_secret = os.environ["SENTINELHUB_CLIENT_SECRET"]
config.sh_token_url = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
config.sh_base_url = "https://sh.dataspace.copernicus.eu"


def download_geotiff(
    bbox_coordinates: Tuple[float, float, float, float] = (46.16, -16.15, 46.51, -15.58),
    download_path: Path = Path("data"),
    time_interval: Tuple[str, str] = ("2020-06-01", "2023-11-02"),
):
    """Download Sentinel-2 data as a GeoTIFF file.

    Args:
        bbox_coordinates (Tuple[float, float, float, float], optional): Bounding box coordinates in WGS84
            projection. Defaults to (46.16, -16.15, 46.51, -15.58).
        download_path (Path, optional): Path to download data to. Defaults to Path("data").
        time_interval (Tuple[str, str], optional): Time interval to download data for. Defaults to
            ("2020-06-01", "2023-11-02").
    """
    resolution = 60
    bbox = BBox(bbox=bbox_coordinates, crs=CRS.WGS84)
    bbox_size = bbox_to_dimensions(bbox, resolution=resolution)

    evalscript_all_bands = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"],
                    units: "DN"
                }],
                output: {
                    bands: 13,
                    sampleType: "INT16"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B01,
                    sample.B02,
                    sample.B03,
                    sample.B04,
                    sample.B05,
                    sample.B06,
                    sample.B07,
                    sample.B08,
                    sample.B8A,
                    sample.B09,
                    sample.B10,
                    sample.B11,
                    sample.B12];
        }
    """

    request_all_bands = SentinelHubRequest(
        data_folder=download_path,
        evalscript=evalscript_all_bands,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C.define_from(
                    "s2l1c", service_url=config.sh_base_url
                ),
                time_interval=time_interval,
                # TODO figure out what mosiacking order does
                mosaicking_order=MosaickingOrder.MOST_RECENT,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=bbox_size,
        config=config,
    )

    request_all_bands.save_data()


if __name__ == "__main__":
    download_geotiff((46.16, -16.15, 46.51, -15.58))
