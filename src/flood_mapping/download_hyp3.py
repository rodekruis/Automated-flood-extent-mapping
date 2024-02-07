import os
import tempfile
import time
from pathlib import Path
from shutil import unpack_archive

import dotenv
import requests

dotenv.load_dotenv()

HYP3_COOKIE = os.environ["HYP3_COOKIE"]


def make_hyp3_request(endpoint, method="GET", **kwargs):
    """Make a request to the HYP3 API with the required cookie.

    To get the cookie:
    1. Log in the HYP3 website:
        https://search.asf.alaska.edu/
    2. Go to the swagger API documentation:
        https://hyp3-api.asf.alaska.edu/ui/
    3. Open the developer tools in the browser and go to the network tab.
    4. Make a request to the GET `/user` endpoint.
    5. Right click the request in the dev tools and select "Copy" -> "Copy as cURL (bash)".
    6. Paste the cURL command in a text editor and copy the part of the command between `asf-urs=` and the next semicolon.

    Args:
        url (str): The URL to make the request to.

    Returns:
        dict: The JSON response from the request.
    """
    cookies = {
        "asf-urs": HYP3_COOKIE,
    }
    url = "https://hyp3-api.asf.alaska.edu/" + endpoint
    response = requests.request(method, url, cookies=cookies, **kwargs)
    return response.json()


def try_user_info():
    endpoint = "user"
    return make_hyp3_request(endpoint)


def run_vv_vh_job(scene_name):
    """Submit a job, wait for it to finish, and download the results."""
    response = submit_vv_vh_job(scene_name)
    job_id = response["jobs"][0]["job_id"]
    response = wait_to_finish(job_id)
    download_url = response["files"][0]["url"]
    download_data(download_url)


def submit_vv_vh_job(scene_name):
    """Submit a job to the HYP3 API to process a Sentinel-1 scene to VV and VH backscatter."""
    data = {
        "jobs": [
            {
                "job_parameters": {
                    "dem_matching": False,
                    "dem_name": "copernicus",
                    "granules": [scene_name],
                    "include_dem": False,
                    "include_inc_map": False,
                    "include_rgb": False,
                    "include_scattering_area": False,
                    "radiometry": "gamma0",
                    "resolution": 20,
                    "scale": "power",
                    "speckle_filter": False,
                },
                "job_type": "RTC_GAMMA",
                "name": f"RTC Job for {scene_name}",
            },
        ],
        "validate_only": False,
    }
    endpoint = "jobs"
    response = make_hyp3_request(endpoint, method="POST", json=data)
    return response


def wait_to_finish(job_id):
    """Wait for a job to finish and return the response."""
    status = "RUNNING"
    while True:
        endpoint = f"jobs/{job_id}"
        response = make_hyp3_request(endpoint)
        try:
            status = response["status_code"]
        except (KeyError, IndexError) as e:
            print("status could not be fetched from:", response)
            raise e
        if status == "RUNNING":
            print("Job is still running")
            time.sleep(60)
        else:
            break
    return response


def download_data(url):
    """Download the processed HYP3 zip (1GB) and extract it to data"""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "download.zip"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with temp_file_path.open("wb") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
        unpack_archive(temp_file_path, "data")


if __name__ == "__main__":
    print(try_user_info())
    scene_name = "S1A_IW_GRDH_1SDV_20231228T173337_20231228T173402_051858_0643CB_C0B6"
