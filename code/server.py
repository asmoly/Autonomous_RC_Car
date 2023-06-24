import socket
 
UDP_IP = "73.254.187.38"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    sock.sendto(b"testing", (UDP_IP, UDP_PORT))