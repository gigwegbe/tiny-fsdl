from mqtt import MQTTClient
from secret import SSID, KEY, user, password, server
import network
import time

KEY_PATH = "newkey.der"
CERT_PATH = "openmv.cert.der"

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


def callback(topic, msg):
    print(topic, msg)


# must set callback first
client.set_callback(callback)
client.subscribe("openmv/test")


# For subscribe
while (True):
    client.wait_msg()  # poll for messages.
    time.sleep_ms(1000)

# For publish
# while (True):
    #client.publish("openmv/test", "Hello World!")
    # time.sleep_ms(1000)


# https://forums.openmv.io/t/aws-mqtt-private-key-form-pem-to-der/5929 - to setup key & cert
