
import time
import socket
import base64
import zlib


class MameCommunicator:
    def __init__(self, host="127.0.0.1", port=12345):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.sock.settimeout(1000) 
        self.number_of_messages=0
        self.debug = False

    def send_to_lua(self, messages):
        message_str = '\n'.join(messages) + '\n__end__\n'
        self.sock.sendall(message_str.encode())
        self.number_of_messages+=len(messages)
        if self.debug:
            print(f"MameCommunicator Send2LUA: {message_str}")


    def receive_from_lua(self):
        try:
            # Réception des données brutes compressées
            data = self.sock.recv(4096)  # Lire jusqu'à 4 KB (ajuster si nécessaire)
            if not data:  # Vérifier si le socket a renvoyé des données vides
                return []
            # Décompresser avec zlib et convertir en texte
            decompressed_data = zlib.decompress(data).decode()
            # Découper les messages
            messages = decompressed_data.split("\n")
            return [msg for msg in messages if msg]  # Supprimer les lignes vides

        except Exception as e:
            print(f"Erreur de réception/décompression: {e}")
            return []

    def communicate(self, messages):
        self.send_to_lua(messages)
        return self.receive_from_lua()

if __name__ == "__main__":
    comm = MameCommunicator(port= 12345)

    # Exemple d'utilisation de la classe MameCommunicator pour envoyer des commandes et lire la mémoire

    speed=0
    # Écriture  d'une valeur de mémoire
    numCoins = "20EB"
    valeur = "10"
    response = comm.communicate([f"write_memory {numCoins}({valeur})"])
    print(response)
    response = comm.communicate(["execute P1_start(1)"])
    print(response)
    for _ in range(1000):
        # Envoie la commande pour déplacer le joueur 1 à gauche
        response = comm.communicate(["execute P1_left(1)","execute P1_Button_1(1)"])
        print (response)
        time.sleep(speed)
        # Envoie la commande pour déplacer le joueur 1 à droite
        response = comm.communicate(["execute P1_left(0)","execute P1_right(1)","execute P1_Button_1(0)"]) 
        print (response)
        time.sleep(speed)
        P1ScorL = "20F8"
        P1ScorM = "20F9"
        P1ScorL_v,P1ScorM_v  =list(map(int,comm.communicate([f"read_memory {P1ScorL}",f"read_memory {P1ScorM}"])))
        score = (
            (P1ScorL_v >> 4) * 10 + (P1ScorM_v & 0x0F) * 100 + ((P1ScorM_v) >> 4) * 1000
        )

        print(f"Score =>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  {score}")
