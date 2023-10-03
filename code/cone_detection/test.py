import base64
from io import BytesIO
from PIL import Image


base64_string = "eJwBQAG//olQTkcNChoKAAAADUlIRFIAAAB2AAAAggEDAAAAqWquXQAAAAZQTFRFAAAA////pdmf3QAAAAF0Uk5TAEDm2GYAAADoSURBVHictdVLDoQgDABQiAuXHMGjeLOBo3EUj8CShbEjjpSCykcdFyYvqS20GBjbngEUo488WDcZip6emAOYRlvqruyZum/38k+LsiHnocKKeGy0PLEueMoYKmwe2l6b3/H8njvn5aHh2v2p1WsWt6zRQ5Wnl23QY5VtteUthwMBefO8d6LFbn9gxt2+IdLb0mphYsEm8ZwYaHmfoCfWtPzPNNyZhrsV0nC3AEgsc7aJIcm3fhDVW3fEE2P7vKMAy+IAnTQgGgB2GNegogGHX1rgdqKRajwRPcmGGSbiIbkxDjc++2zvLwhLCf/ujVLzAAAAAElFTkSuQmCCOMCLDg=="

# Decode the Base64 string
image_data = base64.b64decode(base64_string)

# Convert the binary image data to an image object (assuming it's a PNG image)
image = Image.open(BytesIO(image_data))

# You can display or save the image as needed
image.show()