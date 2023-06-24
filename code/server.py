import socket
 
UDP_IP = "10.42.0.120"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    sock.sendto(b"hello", (UDP_IP, UDP_PORT))
