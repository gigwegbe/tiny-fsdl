# Post files with HTTP/Post urequests module example
import mrequests
import network
from secret import SSID, KEY, user, password, server
import machine
import time

# AP info
SSID = SSID  # Network SSID
KEY = KEY  # Network key


# Init wlan module and connect to network
print("Trying to connect... (may take a while)...")

wlan = network.WINC()
wlan.connect(SSID, key=KEY, security=wlan.WPA_PSK)

# We should have a valid IP now via DHCP
print(wlan.ifconfig())


def reset_device():
    print("Resetting system in 3 seconds.")
    time.sleep(3)
    machine.reset()


def download_model(filename, url):
    headers = {b"accept": b"application/octet-stream"}
    r = mrequests.get(url, headers=headers)

    print(r)
    if r.status_code == 200:
        r.save(filename)
        print("Image saved to '{}'.".format(filename))
    else:
        print("Request failed. Status: {}".format(r.status_code))
    r.close()
    reset_device()


filename = "trained.tflite"
url = f"https://openmv-bucket.s3.us-east-2.amazonaws.com/model/{filename}"

download_model(filename, url)


# other model "https://raw.github.com/gigwegbe/tinyml-digital-counter-for-metering/main/notebook/model/trained.tflite
