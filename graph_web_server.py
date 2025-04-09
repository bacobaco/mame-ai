from flask import Flask, send_from_directory, render_template_string
import os

class GraphWebServer:
    def __init__(self, graph_dir="graphs", host="0.0.0.0", port=5000, auto_display_latest=True):
        """
        graph_dir: dossier où sont sauvegardés les graphiques (doit exister ou sera créé)
        host: adresse d'écoute (0.0.0.0 pour être accessible sur le réseau local)
        port: port d'écoute
        auto_display_latest: si True, affiche directement le dernier PNG en date ; sinon, affiche la liste des fichiers
        """
        self.graph_dir = graph_dir
        self.host = host
        self.port = port
        self.auto_display_latest = auto_display_latest
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            # Crée le dossier s'il n'existe pas
            if not os.path.exists(self.graph_dir):
                os.makedirs(self.graph_dir)
            if self.auto_display_latest:
                # Récupère la liste des fichiers PNG dans le dossier
                files = [f for f in os.listdir(self.graph_dir) if f.lower().endswith(".png")]
                if files:
                    # Sélectionne le fichier le plus récent en fonction de sa date de modification
                    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(self.graph_dir, f)))
                else:
                    latest_file = None
                # Template HTML qui affiche directement le dernier graphique
                html = """
                <!doctype html>
                <html>
                  <head>
                    <meta charset="utf-8">
                    <title>Graphique des parties</title>
                  </head>
                  <body>
                    <h1>Graphique des parties</h1>
                    {% if latest_file %}
                      <img src="/graphs/{{ latest_file }}" alt="Graphique des parties" style="max-width:100%; height:auto;">
                    {% else %}
                      <p>Aucun graphique disponible.</p>
                    {% endif %}
                  </body>
                </html>
                """
                return render_template_string(html, latest_file=latest_file)
            else:
                # Sinon, affiche la liste des fichiers disponibles
                files = [f for f in os.listdir(self.graph_dir) if f.lower().endswith(".png")]
                html = """
                <!doctype html>
                <html>
                  <head>
                    <meta charset="utf-8">
                    <title>Évolution des parties</title>
                  </head>
                  <body>
                    <h1>Graphiques des parties</h1>
                    {% if files %}
                      <ul>
                      {% for file in files %}
                        <li><a href="/graphs/{{ file }}">{{ file }}</a></li>
                      {% endfor %}
                      </ul>
                    {% else %}
                      <p>Aucun graphique disponible.</p>
                    {% endif %}
                  </body>
                </html>
                """
                return render_template_string(html, files=files)

        @self.app.route("/graphs/<path:filename>")
        def serve_graph(filename):
            # Sert le fichier depuis le dossier graph_dir
            return send_from_directory(self.graph_dir, filename)

    def run(self):
        # Démarre le serveur sur l'adresse et le port spécifiés
        self.app.run(host=self.host, port=self.port)
