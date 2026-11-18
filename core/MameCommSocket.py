
import time
import socket
import base64
import zlib


class MameCommunicator:
    def __init__(self, host="127.0.0.1", port=12345, deferred_accept=False, recv_timeout=None):
        self.debug = False
        self.host = host
        self.port = port
        self.sock = None
        self.server_sock = None
        # recv_timeout (secondes) : None = bloquant infini (comportement historique).
        # Si défini, une liaison gelée/coupée (VPN, pare-feu, crash MAME) lève une exception au lieu de bloquer à vie.
        self.recv_timeout = recv_timeout
        
        # MODE SERVEUR UNIQUE (MAME Native Socket est Client)
        print(f"[MAMEComm] MameCommunicator: Démarrage SERVEUR sur {self.host}:{self.port}...")
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_sock.bind((self.host, self.port))
            self.server_sock.listen(1)
            
            if not deferred_accept:
                self.accept_connection()
                
        except OSError as e:
            if e.winerror == 10013: # WSAEACCES
                print(f"[ERROR] ERREUR CRITIQUE: Le port {self.port} est verrouillé par un autre processus (MAME ou Python).")
            raise ConnectionRefusedError(f"[ERROR] Erreur critique serveur : {e}")
        except Exception as e:
            raise ConnectionRefusedError(f"[ERROR] Erreur critique serveur : {e}")
            
        self.number_of_messages=0
 
    def accept_connection(self):
        """Attend et accepte la connexion entrante de MAME."""
        if self.sock: return # Déjà connecté
        
        try:
            print("[WAIT] En attente de la connexion de MAME (Lua)...")
            
            # On accepte la connexion (bloquant, ou limité à 120 s si recv_timeout est défini)
            if self.recv_timeout is not None:
                self.server_sock.settimeout(120)
            self.sock, addr = self.server_sock.accept()
            print(f"[OK] MAME connecté depuis {addr}")
            
            # On ferme le socket d'écoute, on garde la connexion établie
            self.server_sock.close()
            self.server_sock = None
            self.sock.settimeout(self.recv_timeout) # None = mode bloquant pour les échanges
        except Exception as e:
             raise ConnectionRefusedError(f"[ERROR] Erreur lors de l'acceptation de la connexion : {e}")

    def close(self):
        """Ferme proprement les sockets (utilisé lors d'une relance automatique de MAME)."""
        for s in (self.sock, self.server_sock):
            if s is not None:
                try: s.close()
                except Exception: pass
        self.sock = None
        self.server_sock = None


    def send_to_lua(self, messages):
        message_str = '\n'.join(messages) + '\n__end__\n'
        self.sock.sendall(message_str.encode())
        self.number_of_messages+=len(messages)
        if self.debug:
            print(f"MameCommunicator Send2LUA: {message_str}")


    def receive_from_lua(self):
        try:
            # Réception des données brutes compressées
            data = self.sock.recv(32768)  # Lire jusqu'à 32 KB (ajuster si nécessaire)
            if not data:  # Vérifier si le socket a renvoyé des données vides
                if self.recv_timeout is not None:
                    raise ConnectionError("MAME a fermé la connexion")
                return []
            # Décompresser avec zlib et convertir en texte
            decompressed_data = zlib.decompress(data).decode()
            # Découper les messages
            messages = decompressed_data.split("\n")
            return [msg for msg in messages if msg]  # Supprimer les lignes vides

        except (socket.timeout, ConnectionError, OSError):
            if self.recv_timeout is not None:
                raise  # Liaison gelée/coupée : on laisse l'appelant relancer MAME
            print("Erreur de réception : connexion perdue")
            return []
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
