import os

import dotenv
import requests

dotenv.load_dotenv()

HYP3_COOKIE = os.environ["HYP3_COOKIE"]


def make_hyp3_request(url):
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

    response = requests.get(url, cookies=cookies)
    return response.json()


def try_user_info():
    url = "https://hyp3-api.asf.alaska.edu/v1/users/info"
    return make_hyp3_request(url)


if __name__ == "__main__":
    print(try_user_info())
