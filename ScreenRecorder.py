from obswebsocket import obsws, requests
import time
class ScreenRecorder:
    """Evidemment préparez OBS avec la scéne et fenêtre de mame avant prêt à l'enregistrement !
    doc sur les commandes ici: https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md#startrecord
    """

    def __init__(self, host="localhost", port=4455, password="baco22"):
        self.ws = obsws(host, port, password)
        try:
            self.ws.connect()
            print(self.ws.call(requests.GetVersion()).getObsVersion())
        except Exception as e:
            print(f"Erreur lors de la connexion à OBS Studio : {e}")

    def start_recording(self):
        try:
            if not self.ws.call(requests.StartRecord()).status:
                print(
                    "Visiblement l'enregistrement est toujours en cours au moment du StartRecord()..."
                )
        except Exception as e:
            print(f"Erreur lors du démarrage de l'enregistrement : {e}")

    def stop_recording(self):
        try:
            self.ws.call(requests.StopRecord())
        except Exception as e:
            print(f"Erreur lors de l'arrêt de l'enregistrement : {e}")


if __name__ == "__main__":
    recorder = ScreenRecorder()
    recorder.start_recording()

    # Enregistrement pendant 10 secondes
    time.sleep(10)

    # Arrêtez l'enregistrement quand vous avez fini
    recorder.stop_recording()
