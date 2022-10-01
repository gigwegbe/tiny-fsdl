from mqtt import MQTTClient
from secret import SSID, KEY, user, password, server
import network
import time
import json
from secret import SSID, KEY, user, password, server
import json
import ubinascii
import os
import image

KEY_PATH = "newkey.der"
CERT_PATH = "openmv.cert.der"
sample_folder = "single_full"
files_path = [x for x in os.listdir(sample_folder) if x[0] != "."]
print(files_path)

wlan = network.WLAN(network.STA_IF)
# wlan.deinit()
wlan.active(True)
wlan.connect(SSID, KEY, security=wlan.WPA_PSK)

# We should have a valid IP now via DHCP
print("WiFi Connected ", wlan.ifconfig())


with open(KEY_PATH, 'r') as f:
    key1 = f.read()
with open(CERT_PATH, 'r') as f:
    cert1 = f.read()


client = MQTTClient(client_id="openmv",
                    server=server,
                    port=8883,
                    keepalive=4000,
                    ssl=True,
                    ssl_params={"key": key1, "cert": cert1, "server_side": False})
client.connect()
print("MQTT client conencted")


def compress_image(img_full):
    img_full.compress(quality=90)
    img_bytes = ubinascii.b2a_base64(img_full)
    img_t = img_bytes.decode("utf-8")  # img_t is a string
    img_t = img_t.strip()
    print(img_t)
    return img_t


img = image.Image(sample_folder + "/" +
                  str("Img_6_473385909.bmp"), copy_to_fb=True)
comp_img = compress_image(img)

message = {"message": comp_img}
messageJSON = json.dumps(message)

# For publish
while (True):
    client.publish("outTopic", messageJSON)
    time.sleep_ms(1000)


# https://forums.openmv.io/t/aws-mqtt-private-key-form-pem-to-der/5929 - to setup key & cert
# https://stackoverflow.com/questions/66412816/sending-image-as-numpy-array-over-mqtt-to-aws-iot-core
# https://www.youtube.com/watch?v=MsyzeXMu23w&t=624s
