import sys
import time
from obswebsocket import obsws, requests

class ScreenRecorder:
    """Gestionnaire d'enregistrement OBS Studio via obs-websocket.
    Adapte automatiquement la fenêtre cible ('Pac-Man' ou 'Space Invaders')
    selon le jeu exécuté.
    """

    def __init__(self, host="localhost", port=4455, password="baco22", game=None):
        # Détection automatique du jeu selon le nom du script ou argument explicite
        if game is not None:
            self.game = game.lower()
        else:
            script_name = sys.argv[0].lower() if sys.argv else ""
            if "invader" in script_name:
                self.game = "invaders"
            else:
                self.game = "pacman"

        self.ws = obsws(host, port, password)
        try:
            self.ws.connect()
            print(f"OBS Studio connecté ({self.ws.call(requests.GetVersion()).getObsVersion()}). Jeu cible : {self.game.upper()}")
            self.set_target_game(self.game)
        except Exception as e:
            print(f"Erreur lors de la connexion à OBS Studio : {e}")
            self.ws = None # Invalide l'objet en cas d'échec de connexion

    def set_target_game(self, game_name):
        """Configure la capture de fenêtre OBS pour cibler automatiquement le bon jeu MAME."""
        if not self.ws:
            return
        
        # Titres de fenêtres MAME officiels
        if "invader" in game_name.lower():
            window_str = "Space Invaders / Space Invaders M [invaders]:MAME:mame.exe"
        else:
            window_str = "Pac-Man [pacman]:MAME:mame.exe"

        try:
            # priority=2 : cherche le titre exact, et sinon capture n'importe quelle fenêtre de mame.exe
            self.ws.call(requests.SetInputSettings(
                inputName='Capture de fenêtre',
                inputSettings={'priority': 2, 'window': window_str},
                overlay=True
            ))
        except Exception as e:
            # Si le nom de la source dans OBS est légèrement différent, on ne bloque pas
            pass

    def get_recording_status(self):
        """Vérifie si OBS est en train d'enregistrer."""
        if not self.ws: return False
        try:
            status = self.ws.call(requests.GetRecordStatus())
            return status.getOutputActive()
        except Exception:
            return False

    def start_recording(self):
        """Démarre l'enregistrement, en s'assurant qu'aucun n'est déjà en cours."""
        if not self.ws: return
        try:
            if self.get_recording_status():
                print("⚠️ WARNING! OBS est déjà en train d'enregistrer. Tentative d'arrêt avant de redémarrer.")
                self.stop_recording() # Tente d'arrêter l'enregistrement précédent
                time.sleep(1) # Laisse à OBS le temps de finaliser le fichier

            # On redémarre l'enregistrement
            if not self.ws.call(requests.StartRecord()).status:
                print("❌ ERROR! Impossible de démarrer l'enregistrement OBS après tentative de réinitialisation.")
        except Exception as e:
            print(f"❌ Erreur lors du démarrage de l'enregistrement : {e}")

    def stop_recording(self):
        """Arrête l'enregistrement et retourne le chemin du fichier vidéo."""
        if not self.ws: return None
        try:
            if self.get_recording_status():
                response = self.ws.call(requests.StopRecord())
                # La réponse contient le chemin du fichier enregistré
                video_path = response.getOutputPath()
                
                # Attendre que OBS confirme l'arrêt (max 3s)
                for _ in range(15):
                    time.sleep(0.2)
                    if not self.get_recording_status(): break
                
                return video_path
            else:
                return None
        except Exception as e:
            print(f"❌ Erreur lors de l'arrêt de l'enregistrement : {e}")
            return None


if __name__ == "__main__":
    recorder = ScreenRecorder()
    recorder.start_recording()

    # Enregistrement pendant 10 secondes
    time.sleep(1)
    path = recorder.stop_recording()
    time.sleep(1)
    print("Arrêt de l'enregistrement... path=%s" % path)
