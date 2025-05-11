import pygame
import sys
import random
import math

# Initialisation de Pygame
pygame.init()

# Couleurs élégantes
FOND = (240, 248, 255)  # Bleu très clair
NOIR = (40, 44, 52)  # Noir profond
ROUGE = (220, 76, 70)  # Rouge vif mais élégant
BLEU = (66, 139, 202)  # Bleu élégant
GRIS = (180, 180, 180)  # Gris clair
VERT = (92, 184, 92)  # Vert élégant
JAUNE_PALE = (255, 248, 225)  # Jaune très pâle
ROUGE_PALE = (255, 235, 235)  # Rouge très pâle
BLEU_PALE = (235, 245, 255)  # Bleu très pâle
OR = (212, 175, 55)  # Couleur or
VIOLET = (153, 102, 204)  # Violet pour slider

# Tailles de grille disponibles
TAILLES_GRILLE = {
    "Petite": 5,
    "Moyenne": 7,
    "Grande": 9
}

class Slider:
    def __init__(self, x, y, largeur, hauteur, min_val, max_val, valeur_initiale):
        self.rect = pygame.Rect(x, y, largeur, hauteur)
        self.knob_rect = pygame.Rect(0, 0, 20, hauteur + 10)
        self.min_val = min_val
        self.max_val = max_val
        self.valeur = valeur_initiale
        self.en_glissement = False
        self.couleur_barre = GRIS
        self.couleur_knob = VIOLET
        self.mettre_a_jour_knob()
        self.etiquettes = ["Facile", "Moyen", "Difficile"]

    def mettre_a_jour_knob(self):
        # Calculer la position du bouton
        position_relative = (self.valeur - self.min_val) / (self.max_val - self.min_val)
        self.knob_rect.centerx = int(self.rect.left + position_relative * self.rect.width)
        self.knob_rect.centery = self.rect.centery

    def dessiner(self, fenetre):
        # Dessiner le fond du slider
        pygame.draw.rect(fenetre, self.couleur_barre, self.rect, border_radius=5)
        pygame.draw.rect(fenetre, NOIR, self.rect, 2, border_radius=5)
        
        # Dessiner le bouton du slider
        pygame.draw.rect(fenetre, self.couleur_knob, self.knob_rect, border_radius=10)
        pygame.draw.rect(fenetre, NOIR, self.knob_rect, 2, border_radius=10)
        
        # Afficher les étiquettes
        font = pygame.font.SysFont(None, 20)
        
        # Dessiner des graduations
        for i in range(3):
            pos_x = self.rect.left + (i * self.rect.width / 2)
            pygame.draw.line(fenetre, NOIR, (pos_x, self.rect.bottom), (pos_x, self.rect.bottom + 5), 2)

            etiquette = font.render(self.etiquettes[i], True, NOIR)
            fenetre.blit(etiquette, (pos_x - etiquette.get_width() // 2, self.rect.bottom + 10))

    def gerer_evenement(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.knob_rect.collidepoint(event.pos):
                self.en_glissement = True
        
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.en_glissement = False
        
        elif event.type == pygame.MOUSEMOTION and self.en_glissement:
            # Limiter aux bords du slider
            rel_x = min(max(event.pos[0], self.rect.left), self.rect.right)
            position_relative = (rel_x - self.rect.left) / self.rect.width
            self.valeur = self.min_val + position_relative * (self.max_val - self.min_val)
            self.mettre_a_jour_knob()
            return True  # Valeur modifiée
        
        return False  # Pas de changement

class BoutonTaille:
    def __init__(self, x, y, largeur, hauteur, texte, taille):
        self.rect = pygame.Rect(x, y, largeur, hauteur)
        self.texte = texte
        self.taille = taille
        self.survol = False

    def dessiner(self, fenetre):
        couleur = GRIS
        if self.survol:
            couleur = (150, 150, 150)
        pygame.draw.rect(fenetre, couleur, self.rect, border_radius=10)
        pygame.draw.rect(fenetre, NOIR, self.rect, 2, border_radius=10)
        
        font = pygame.font.SysFont(None, 24)
        texte_surface = font.render(self.texte, True, NOIR)
        fenetre.blit(texte_surface, (
            self.rect.x + (self.rect.width - texte_surface.get_width()) // 2,
            self.rect.y + (self.rect.height - texte_surface.get_height()) // 2
        ))

    def verifier_survol(self, pos):
        self.survol = self.rect.collidepoint(pos)
        return self.survol

    def verifier_clic(self, pos):
        return self.rect.collidepoint(pos)

class Bouton:
    def __init__(self, x, y, largeur, hauteur, texte, couleur=None):
        self.rect = pygame.Rect(x, y, largeur, hauteur)
        self.texte = texte
        self.survol = False
        self.couleur = couleur or GRIS
        self.couleur_survol = (min(self.couleur[0] + 30, 255), 
                              min(self.couleur[1] + 30, 255), 
                              min(self.couleur[2] + 30, 255))

    def dessiner(self, fenetre):
        couleur = self.couleur_survol if self.survol else self.couleur
        pygame.draw.rect(fenetre, couleur, self.rect, border_radius=10)
        pygame.draw.rect(fenetre, NOIR, self.rect, 2, border_radius=10)
        
        font = pygame.font.SysFont(None, 24)
        texte_surface = font.render(self.texte, True, NOIR)
        fenetre.blit(texte_surface, (
            self.rect.x + (self.rect.width - texte_surface.get_width()) // 2,
            self.rect.y + (self.rect.height - texte_surface.get_height()) // 2
        ))

    def verifier_survol(self, pos):
        self.survol = self.rect.collidepoint(pos)
        return self.survol

    def verifier_clic(self, pos):
        return self.rect.collidepoint(pos)

class JeuPipopinette:
    def __init__(self, taille_grille, difficulte_ia=0.5):
        self.taille_grille = taille_grille
        self.difficulte_ia = difficulte_ia
        
        # Ajustement de la taille de la fenêtre en fonction de la taille de la grille
        if taille_grille <= 5:
            self.largeur_fenetre = 1000
            self.hauteur_fenetre = 700
        else:
            self.largeur_fenetre = 1200
            self.hauteur_fenetre = 800
            
        self.fenetre = pygame.display.set_mode((self.largeur_fenetre, self.hauteur_fenetre))
        pygame.display.set_caption("Pipopinette - Le Jeu de Points")
        self.horloge = pygame.time.Clock()
        
        # Chargement des polices d'écriture
        pygame.font.init()
        
        try:
            self.police_titre = pygame.font.SysFont('segoeuiemoji', 40, bold=True)
            self.police = pygame.font.SysFont('segoeuiemoji', 24)
            self.police_score = pygame.font.SysFont('segoeuiemoji', 28, bold=True)
            self.police_info = pygame.font.SysFont('segoeuiemoji', 10)
        except:
            self.police_titre = pygame.font.SysFont(None, 36, bold=True)
            self.police = pygame.font.SysFont(None, 24)
            self.police_score = pygame.font.SysFont(None, 28, bold=True)
            self.police_info = pygame.font.SysFont(None, 10)
        
        # Dimensions pour le panneau latéral d'information
        self.panneau_lateral_largeur = self.largeur_fenetre * 0.25
        
        # Centrage et taille de la grille
        plateau_size = min((self.largeur_fenetre - self.panneau_lateral_largeur) * 0.85, self.hauteur_fenetre * 0.7)
        self.marge_horizontale = (self.largeur_fenetre - self.panneau_lateral_largeur - plateau_size) // 2
        self.marge_verticale = (self.hauteur_fenetre - plateau_size) // 2 + 30  # Ajustement pour laisser de la place au titre
        self.taille_case = plateau_size // (self.taille_grille - 1)
        
        # Initialisation des lignes (0 = pas de ligne, 1 = ligne du joueur, 2 = ligne de l'ordinateur)
        self.lignes_horizontales = [[0 for _ in range(self.taille_grille - 1)] for _ in range(self.taille_grille)]
        self.lignes_verticales = [[0 for _ in range(self.taille_grille)] for _ in range(self.taille_grille - 1)]
        
        # Initialisation des cases (0 = pas de propriétaire, 1 = joueur, 2 = ordinateur)
        self.cases = [[0 for _ in range(self.taille_grille - 1)] for _ in range(self.taille_grille - 1)]
        
        # Score
        self.score_joueur = 0
        self.score_ordi = 0
        
        # Tour (1 = joueur, 2 = ordinateur)
        self.tour = 1
        
        # Partie terminée ?
        self.jeu_termine = False
        
        # Message de fin
        self.message_fin = ""
        
        # Animation
        self.animation_alpha = 0
        self.animation_direction = 1
        
        # Effets de particules
        self.particules = []
        
        # Feedback visuel
        self.derniere_ligne = None
        self.temps_derniere_ligne = 0
        
        # Créer le slider pour la difficulté de l'IA
        self.slider_difficulte = Slider(
            self.largeur_fenetre - self.panneau_lateral_largeur + 50,
            self.hauteur_fenetre // 2 + 50,
            self.panneau_lateral_largeur - 100,
            20,
            0.0,
            1.0,
            self.difficulte_ia
        )
        
        y_bas = self.hauteur_fenetre//2 +200

        self.bouton_recommencer = Bouton(
            self.largeur_fenetre - self.panneau_lateral_largeur + 50,
            y_bas -50,
            self.panneau_lateral_largeur - 100,
            40,
            "Nouvelle Partie",
            VERT
        )

        self.bouton_menu = Bouton(
            self.largeur_fenetre - self.panneau_lateral_largeur + 50,
            y_bas - 100,
            self.panneau_lateral_largeur - 100,
            40,
            "Menu Principal",
            GRIS
        )
        
    def creer_particules(self, x, y, couleur, nombre=15):
        """Crée des particules pour les effets visuels"""
        for _ in range(nombre):
            angle = random.uniform(0, 2 * math.pi)
            vitesse = random.uniform(1, 4)
            duree_vie = random.randint(20, 40)
            taille = random.uniform(2, 6)
            self.particules.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * vitesse,
                'vy': math.sin(angle) * vitesse,
                'couleur': couleur,
                'duree_vie': duree_vie,
                'taille': taille
            })
    
    def mettre_a_jour_particules(self):
        """Met à jour les particules et les supprime si nécessaire"""
        for i in range(len(self.particules) - 1, -1, -1):
            p = self.particules[i]
            p['duree_vie'] -= 1
            if p['duree_vie'] <= 0:
                self.particules.pop(i)
                continue
            
            p['x'] += p['vx']
            p['y'] += p['vy']
            # Effet de réduction progressive
            p['taille'] *= 0.95
    
    def dessiner_panneau_lateral(self):
        """Dessiner le panneau d'information à droite"""
        # Zone du panneau
        panneau_rect = pygame.Rect(
            self.largeur_fenetre - self.panneau_lateral_largeur,
            0,
            self.panneau_lateral_largeur,
            self.hauteur_fenetre
        )
        
        # Fond du panneau
        pygame.draw.rect(self.fenetre, (245, 245, 245), panneau_rect)
        pygame.draw.line(self.fenetre, GRIS, 
                         (panneau_rect.left, 0), 
                         (panneau_rect.left, self.hauteur_fenetre), 
                         3)
        

        
        # Scores détaillés
        y_pos = 30
        titre_score = self.police_titre.render("Scores", True, NOIR)
        self.fenetre.blit(titre_score, (
            panneau_rect.centerx - titre_score.get_width() // 2,
            y_pos
        ))
        
        # Score du joueur
        y_pos += 50
        score_joueur_rect = pygame.Rect(
            panneau_rect.left + 20,
            y_pos,
            panneau_rect.width - 40,
            50
        )
        pygame.draw.rect(self.fenetre, ROUGE_PALE, score_joueur_rect, border_radius=10)
        pygame.draw.rect(self.fenetre, ROUGE, score_joueur_rect, 3, border_radius=10)
        
        score_joueur_txt = self.police_score.render(f"Vous: {self.score_joueur}", True, ROUGE)
        self.fenetre.blit(score_joueur_txt, (
            score_joueur_rect.centerx - score_joueur_txt.get_width() // 2,
            score_joueur_rect.centery - score_joueur_txt.get_height() // 2
        ))
        
        # Score de l'ordinateur
        y_pos += 70
        score_ordi_rect = pygame.Rect(
            panneau_rect.left + 20,
            y_pos,
            panneau_rect.width - 40,
            50
        )
        pygame.draw.rect(self.fenetre, BLEU_PALE, score_ordi_rect, border_radius=10)
        pygame.draw.rect(self.fenetre, BLEU, score_ordi_rect, 3, border_radius=10)
        
        score_ordi_txt = self.police_score.render(f"Ordi: {self.score_ordi}", True, BLEU)
        self.fenetre.blit(score_ordi_txt, (
            score_ordi_rect.centerx - score_ordi_txt.get_width() // 2,
            score_ordi_rect.centery - score_ordi_txt.get_height() // 2
        ))
        
        # Difficulté de l'IA
        y_pos += 100
        difficulte_titre = self.police.render("Difficulté de l'IA", True, NOIR)
        self.fenetre.blit(difficulte_titre, (
            panneau_rect.centerx - difficulte_titre.get_width() // 2,
            y_pos
        ))

        y_pos += 40
        self.slider_difficulte.rect.y = y_pos
        self.slider_difficulte.mettre_a_jour_knob()
        self.slider_difficulte.dessiner(self.fenetre)

        y_pos=self.hauteur_fenetre
        # Information sur la règle du jeu
        regles_titre = self.police.render("Règles", True, NOIR)
        self.fenetre.blit(regles_titre, (
            panneau_rect.centerx - regles_titre.get_width() // 2,
            y_pos-80
        ))
        
        regles_texte = [
            "• Reliez les points par des lignes",
            "• Complétez des carrés pour marquer des points",
            "• Jouez à nouveau après avoir complété un carré",
            "• Gagnez en obtenant le plus de carrés"
        ]
        
        for ligne in regles_texte:
            texte = self.police_info.render(ligne, True, NOIR)
            self.fenetre.blit(texte, (
                panneau_rect.left + 30,
                y_pos-50
            ))
            y_pos += 1 * texte.get_height()
        
        # Dessiner les boutons
        self.bouton_recommencer.dessiner(self.fenetre)
        self.bouton_menu.dessiner(self.fenetre)
    
    def dessiner_grille(self):
        # Mettre à jour l'animation
        self.animation_alpha += 0.03 * self.animation_direction
        if self.animation_alpha > 1 or self.animation_alpha < 0:
            self.animation_direction *= -1
            self.animation_alpha = max(0, min(1, self.animation_alpha))
        
        # Effacer l'écran
        self.fenetre.fill(FOND)
        
        # Dessiner le panneau latéral
        self.dessiner_panneau_lateral()
        
        # Dessiner un arrière-plan pour le plateau
        plateau_x = self.marge_horizontale - 20
        plateau_y = self.marge_verticale - 20
        plateau_largeur = (self.taille_grille - 1) * self.taille_case + 40
        plateau_hauteur = (self.taille_grille - 1) * self.taille_case + 40
        pygame.draw.rect(self.fenetre, JAUNE_PALE, 
                        (plateau_x, plateau_y, plateau_largeur, plateau_hauteur), 
                        border_radius=15)
        
        # Dessiner le titre du jeu
        titre_surface = self.police_titre.render("Pipopinette", True, NOIR)
        sous_titre_surface = self.police.render("Le Jeu de Points et de Carrés", True, (80, 80, 80))
        self.fenetre.blit(titre_surface, (
            (self.largeur_fenetre - self.panneau_lateral_largeur) // 2 - titre_surface.get_width() // 2,
            15
        ))
        self.fenetre.blit(sous_titre_surface, (
            (self.largeur_fenetre - self.panneau_lateral_largeur) // 2 - sous_titre_surface.get_width() // 2,
            55
        ))
        
        # Dessiner les points avec effet de brillance
        for i in range(self.taille_grille):
            for j in range(self.taille_grille):
                x = self.marge_horizontale + j * self.taille_case
                y = self.marge_verticale + i * self.taille_case
                
                # Point principal
                pygame.draw.circle(self.fenetre, NOIR, (x, y), 7)
                
                # Effet de brillance
                brillance = int(200 + 55 * self.animation_alpha)
                pygame.draw.circle(self.fenetre, (brillance, brillance, brillance), (x, y), 3)
        
        # Dessiner les lignes horizontales
        for i in range(self.taille_grille):
            for j in range(self.taille_grille - 1):
                x1 = self.marge_horizontale + j * self.taille_case
                y1 = self.marge_verticale + i * self.taille_case
                x2 = self.marge_horizontale + (j + 1) * self.taille_case
                y2 = y1
                
                if self.lignes_horizontales[i][j] == 1:
                    # Effet d'ombre
                    pygame.draw.line(self.fenetre, (ROUGE[0]//2, ROUGE[1]//2, ROUGE[2]//2), 
                                    (x1+2, y1+2), (x2+2, y2+2), 5)
                    pygame.draw.line(self.fenetre, ROUGE, (x1, y1), (x2, y2), 5)
                elif self.lignes_horizontales[i][j] == 2:
                    # Effet d'ombre
                    pygame.draw.line(self.fenetre, (BLEU[0]//2, BLEU[1]//2, BLEU[2]//2), 
                                    (x1+2, y1+2), (x2+2, y2+2), 5)
                    pygame.draw.line(self.fenetre, BLEU, (x1, y1), (x2, y2), 5)
                elif self.est_survole_h(i, j):
                    # Ligne en pointillés avec animation pour le survol
                    dash_length = 10
                    gap_length = 5
                    total_length = dash_length + gap_length
                    distance = x2 - x1
                    num_dashes = int(distance / total_length)
                    offset = (self.animation_alpha * total_length) % total_length
                    
                    for k in range(num_dashes + 1):
                        start_x = x1 + offset + k * total_length
                        end_x = min(start_x + dash_length, x2)
                        if start_x < x2:
                            pygame.draw.line(self.fenetre, GRIS, (start_x, y1), (end_x, y1), 3)
        
        # Dessiner les lignes verticales
        for i in range(self.taille_grille - 1):
            for j in range(self.taille_grille):
                x1 = self.marge_horizontale + j * self.taille_case
                y1 = self.marge_verticale + i * self.taille_case
                x2 = x1
                y2 = self.marge_verticale + (i + 1) * self.taille_case
                
                if self.lignes_verticales[i][j] == 1:
                    # Effet d'ombre
                    pygame.draw.line(self.fenetre, (ROUGE[0]//2, ROUGE[1]//2, ROUGE[2]//2), 
                                    (x1+2, y1+2), (x2+2, y2+2), 5)
                    pygame.draw.line(self.fenetre, ROUGE, (x1, y1), (x2, y2), 5)
                elif self.lignes_verticales[i][j] == 2:
                    # Effet d'ombre
                    pygame.draw.line(self.fenetre, (BLEU[0]//2, BLEU[1]//2, BLEU[2]//2), 
                                    (x1+2, y1+2), (x2+2, y2+2), 5)
                    pygame.draw.line(self.fenetre, BLEU, (x1, y1), (x2, y2), 5)
                elif self.est_survole_v(i, j):
                    # Ligne en pointillés avec animation pour le survol
                    dash_length = 10
                    gap_length = 5
                    total_length = dash_length + gap_length
                    distance = y2 - y1
                    num_dashes = int(distance / total_length)
                    offset = (self.animation_alpha * total_length) % total_length
                    
                    for k in range(num_dashes + 1):
                        start_y = y1 + offset + k * total_length
                        end_y = min(start_y + dash_length, y2)
                        if start_y < y2:
                            pygame.draw.line(self.fenetre, GRIS, (x1, start_y), (x1, end_y), 3)
        
        # Remplir les cases complètes avec des effets visuels
        for i in range(self.taille_grille - 1):
            for j in range(self.taille_grille - 1):
                x = self.marge_horizontale + j * self.taille_case
                y = self.marge_verticale + i * self.taille_case
                
                if self.cases[i][j] == 1:
                    # Case du joueur avec dégradé
                    rect = pygame.Rect(x + 5, y + 5, self.taille_case - 10, self.taille_case - 10)
                    pygame.draw.rect(self.fenetre, ROUGE_PALE, rect, border_radius=5)
                    
                    # Ajouter un petit "J" dans la case
                    label = self.police.render("You !", True, ROUGE)
                    self.fenetre.blit(label, 
                        (x + self.taille_case // 2 - label.get_width() // 2,
                         y + self.taille_case // 2 - label.get_height() // 2))
                    
                elif self.cases[i][j] == 2:
                    # Case de l'ordinateur avec dégradé
                    rect = pygame.Rect(x + 5, y + 5, self.taille_case - 10, self.taille_case - 10)
                    pygame.draw.rect(self.fenetre, BLEU_PALE, rect, border_radius=5)
                    
                    # Ajouter un petit "O" dans la case
                    label = self.police.render("Ordi !", True, BLEU)
                    self.fenetre.blit(label, 
                        (x + self.taille_case // 2 - label.get_width() // 2,
                         y + self.taille_case // 2 - label.get_height() // 2))
        
        # Dessiner les particules
        self.mettre_a_jour_particules()
        for p in self.particules:
            pygame.draw.circle(self.fenetre, p['couleur'], 
                              (int(p['x']), int(p['y'])), 
                              int(p['taille']))
        
        # Afficher à qui est le tour dans un joli cadre
        if not self.jeu_termine:
            if self.tour == 1:
                tour_text = "À votre tour"
                couleur_cadre = ROUGE_PALE
                couleur_bordure = ROUGE
                couleur_texte = ROUGE
            else:
                tour_text = "L'ordinateur réfléchit..."
                couleur_cadre = BLEU_PALE
                couleur_bordure = BLEU
                couleur_texte = BLEU
            
            tour_surface = self.police.render(tour_text, True, couleur_texte)
            cadre_tour = pygame.Rect(
                (self.largeur_fenetre - self.panneau_lateral_largeur) // 2 - tour_surface.get_width() // 2 - 20,
                self.hauteur_fenetre - 70,
                tour_surface.get_width() + 40,
                50
            )
            
            pygame.draw.rect(self.fenetre, couleur_cadre, cadre_tour, border_radius=10)
            pygame.draw.rect(self.fenetre, couleur_bordure, cadre_tour, 3, border_radius=10)
            self.fenetre.blit(tour_surface, 
                             ((self.largeur_fenetre - self.panneau_lateral_largeur) // 2 - tour_surface.get_width() // 2,
                              self.hauteur_fenetre - 55))
        else:
            # Message de fin avec un effet d'or brillant
            if "gagné" in self.message_fin:
                couleur_fin = OR
                couleur_cadre = (255, 250, 205)  # Jaune très pâle
                # Couronne de victoire
                victoire_icon = "👑 👍"
            else:
                couleur_fin = NOIR
                couleur_cadre = GRIS
                victoire_icon = "😭 "
            
            fin_surface = self.police_titre.render(victoire_icon + self.message_fin, True, couleur_fin)
            cadre_fin = pygame.Rect(
                (self.largeur_fenetre - self.panneau_lateral_largeur) // 2 - fin_surface.get_width() // 2 - 25,
                self.hauteur_fenetre // 2 - 30,
                fin_surface.get_width() + 50,
                70
            )
            
            pygame.draw.rect(self.fenetre, couleur_cadre, cadre_fin, border_radius=15)
            pygame.draw.rect(self.fenetre, OR, cadre_fin, 4, border_radius=15)
            self.fenetre.blit(fin_surface, 
                             ((self.largeur_fenetre - self.panneau_lateral_largeur) // 2 - fin_surface.get_width() // 2,
                              self.hauteur_fenetre // 2 - 20))
            
    def est_survole_h(self, i, j):
        """Vérifie si la souris survole une ligne horizontale"""
        if self.lignes_horizontales[i][j] != 0:
            return False
            
        pos = pygame.mouse.get_pos()
        x1 = self.marge_horizontale + j * self.taille_case
        y1 = self.marge_verticale + i * self.taille_case
        x2 = self.marge_horizontale + (j + 1) * self.taille_case
        
        # Zone de détection autour de la ligne
        zone = 15  # Zone plus large pour une meilleure interaction
        return (x1 - zone <= pos[0] <= x2 + zone and 
                y1 - zone <= pos[1] <= y1 + zone)
    
    def est_survole_v(self, i, j):
        """Vérifie si la souris survole une ligne verticale"""
        if self.lignes_verticales[i][j] != 0:
            return False
            
        pos = pygame.mouse.get_pos()
        x1 = self.marge_horizontale + j * self.taille_case
        y1 = self.marge_verticale + i * self.taille_case
        y2 = self.marge_verticale + (i + 1) * self.taille_case
        
        # Zone de détection autour de la ligne
        zone = 15  # Zone plus large pour une meilleure interaction
        return (x1 - zone <= pos[0] <= x1 + zone and 
                y1 - zone <= pos[1] <= y2 + zone)
    
    def obtenir_ligne_sous_souris(self):
        """Retourne la ligne (h/v, i, j) sous la souris, ou None"""
        # Vérifier les lignes horizontales
        for i in range(self.taille_grille):
            for j in range(self.taille_grille - 1):
                if self.est_survole_h(i, j) and self.lignes_horizontales[i][j] == 0:
                    return ('h', i, j)
        
        # Vérifier les lignes verticales
        for i in range(self.taille_grille - 1):
            for j in range(self.taille_grille):
                if self.est_survole_v(i, j) and self.lignes_verticales[i][j] == 0:
                    return ('v', i, j)
        
        return None
    
    def verifier_case_complete(self, i, j):
        """Vérifie si une case est complète et l'attribue au joueur qui a fermé la case"""
        if i < 0 or j < 0 or i >= self.taille_grille - 1 or j >= self.taille_grille - 1:
            return False
            
        haut = self.lignes_horizontales[i][j]
        bas = self.lignes_horizontales[i+1][j]
        gauche = self.lignes_verticales[i][j]
        droite = self.lignes_verticales[i][j+1]
        
        if haut != 0 and bas != 0 and gauche != 0 and droite != 0:
            self.cases[i][j] = self.tour
            if self.tour == 1:
                self.score_joueur += 1
                # Effet de particules pour la capture du joueur
                centre_x = self.marge_horizontale + j * self.taille_case + self.taille_case // 2
                centre_y = self.marge_verticale + i * self.taille_case + self.taille_case // 2
                self.creer_particules(centre_x, centre_y, ROUGE)
            else:
                self.score_ordi += 1
                # Effet de particules pour la capture de l'ordinateur
                centre_x = self.marge_horizontale + j * self.taille_case + self.taille_case // 2
                centre_y = self.marge_verticale + i * self.taille_case + self.taille_case // 2
                self.creer_particules(centre_x, centre_y, BLEU)
            return True
        
        return False
    
    def jouer_coup(self, type_ligne, i, j):
        """Joue un coup et vérifie s'il y a des cases complètes"""
        case_completee = False
        
        if type_ligne == 'h':
            self.lignes_horizontales[i][j] = self.tour
            
            # Vérifier si des cases sont complétées
            if i > 0:
                case_completee |= self.verifier_case_complete(i-1, j)
            if i < self.taille_grille - 1:
                case_completee |= self.verifier_case_complete(i, j)
                
        elif type_ligne == 'v':
            self.lignes_verticales[i][j] = self.tour
            
            # Vérifier si des cases sont complétées
            if j > 0:
                case_completee |= self.verifier_case_complete(i, j-1)
            if j < self.taille_grille - 1:
                case_completee |= self.verifier_case_complete(i, j)
        
        # Vérifier si le jeu est terminé
        total_cases = (self.taille_grille - 1) * (self.taille_grille - 1)
        if self.score_joueur + self.score_ordi == total_cases:
            self.jeu_termine = True
            if self.score_joueur > self.score_ordi:
                self.message_fin = f"Vous avez gagné! {self.score_joueur} - {self.score_ordi}"
            elif self.score_ordi > self.score_joueur:
                self.message_fin = f"Vous avez perdu... {self.score_ordi} - {self.score_joueur}"
            else:
                self.message_fin = f"Match nul! {self.score_joueur} - {self.score_ordi}"
        
        # Changer de tour si aucune case n'a été complétée
        if not case_completee and not self.jeu_termine:
            self.tour = 3 - self.tour  # 1 -> 2, 2 -> 1
            
        return case_completee
    
    def coup_ordinateur(self):
        """Intelligence artificielle de l'ordinateur"""
        # Niveau de difficulté de l'IA (affecte la prise de décision)
        niveau_difficulte = self.slider_difficulte.valeur
        
        # Si niveau de difficulté est bas, parfois faire un coup aléatoire
        if random.random() > niveau_difficulte:
            return self.coup_aleatoire()
            
        # Stratégie: compléter les cases à 3 côtés d'abord
        for i in range(self.taille_grille - 1):
            for j in range(self.taille_grille - 1):
                # Compter les côtés de la case
                haut = self.lignes_horizontales[i][j]
                bas = self.lignes_horizontales[i+1][j]
                gauche = self.lignes_verticales[i][j]
                droite = self.lignes_verticales[i][j+1]
                
                cotes = [haut, bas, gauche, droite]
                if cotes.count(0) == 1:  # S'il n'y a qu'un côté manquant
                    # Trouver et jouer le coup manquant
                    if haut == 0:
                        return self.jouer_coup('h', i, j)
                    elif bas == 0:
                        return self.jouer_coup('h', i+1, j)
                    elif gauche == 0:
                        return self.jouer_coup('v', i, j)
                    elif droite == 0:
                        return self.jouer_coup('v', i, j+1)
        
        # Éviter de jouer les cases à 2 côtés (qui donneraient un avantage à l'adversaire)
        # Seulement si le niveau de difficulté est suffisamment élevé
        if niveau_difficulte > 0.3:
            lignes_eviter = []
            for i in range(self.taille_grille - 1):
                for j in range(self.taille_grille - 1):
                    haut = self.lignes_horizontales[i][j]
                    bas = self.lignes_horizontales[i+1][j]
                    gauche = self.lignes_verticales[i][j]
                    droite = self.lignes_verticales[i][j+1]
                    
                    cotes = [haut, bas, gauche, droite]
                    if cotes.count(0) == 2:  # S'il y a deux côtés manquants
                        if haut == 0:
                            lignes_eviter.append(('h', i, j))
                        if bas == 0:
                            lignes_eviter.append(('h', i+1, j))
                        if gauche == 0:
                            lignes_eviter.append(('v', i, j))
                        if droite == 0:
                            lignes_eviter.append(('v', i, j+1))
            
            # Jouer un coup aléatoire parmi ceux qui ne sont pas à éviter
            coups_possibles = []
            
            # Lignes horizontales
            for i in range(self.taille_grille):
                for j in range(self.taille_grille - 1):
                    if self.lignes_horizontales[i][j] == 0:
                        coup = ('h', i, j)
                        if coup not in lignes_eviter:
                            coups_possibles.append(coup)
            
            # Lignes verticales
            for i in range(self.taille_grille - 1):
                for j in range(self.taille_grille):
                    if self.lignes_verticales[i][j] == 0:
                        coup = ('v', i, j)
                        if coup not in lignes_eviter:
                            coups_possibles.append(coup)
            
            # Si on a trouvé des coups non risqués
            if coups_possibles:
                type_ligne, i, j = random.choice(coups_possibles)
                return self.jouer_coup(type_ligne, i, j)
        
        # Si tous les coups sont à éviter ou difficulté faible, on prend n'importe quel coup
        return self.coup_aleatoire()
    
    def coup_aleatoire(self):
        """Joue un coup aléatoire"""
        coups_possibles = []
        
        # Lignes horizontales
        for i in range(self.taille_grille):
            for j in range(self.taille_grille - 1):
                if self.lignes_horizontales[i][j] == 0:
                    coups_possibles.append(('h', i, j))
        
        # Lignes verticales
        for i in range(self.taille_grille - 1):
            for j in range(self.taille_grille):
                if self.lignes_verticales[i][j] == 0:
                    coups_possibles.append(('v', i, j))
        
        if coups_possibles:
            type_ligne, i, j = random.choice(coups_possibles)
            return self.jouer_coup(type_ligne, i, j)
        
        return False
    
    def reinitialiser_jeu(self):
        """Réinitialise le jeu pour une nouvelle partie"""
        self.lignes_horizontales = [[0 for _ in range(self.taille_grille - 1)] for _ in range(self.taille_grille)]
        self.lignes_verticales = [[0 for _ in range(self.taille_grille)] for _ in range(self.taille_grille - 1)]
        self.cases = [[0 for _ in range(self.taille_grille - 1)] for _ in range(self.taille_grille - 1)]
        self.score_joueur = 0
        self.score_ordi = 0
        self.tour = 1
        self.jeu_termine = False
        self.message_fin = ""
        self.particules = []
    
    def executer(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                # Gestion des boutons
                if event.type == pygame.MOUSEMOTION:
                    pos = pygame.mouse.get_pos()
                    self.bouton_recommencer.verifier_survol(pos)
                    self.bouton_menu.verifier_survol(pos)
                
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = pygame.mouse.get_pos()
                    
                    # Gestion du slider de difficulté
                    if self.slider_difficulte.gerer_evenement(event):
                        # La valeur du slider a été modifiée
                        self.difficulte_ia = self.slider_difficulte.valeur
                    
                    # Gestion du bouton recommencer
                    if self.bouton_recommencer.verifier_clic(pos):
                        self.reinitialiser_jeu()
                        continue
                    
                    # Gestion du bouton menu
                    if self.bouton_menu.verifier_clic(pos):
                        return "menu"  # Retourner au menu principal
                    
                    # Gestion du clic sur une ligne (seulement pendant le tour du joueur)
                    if not self.jeu_termine and self.tour == 1:
                        ligne = self.obtenir_ligne_sous_souris()
                        if ligne:
                            type_ligne, i, j = ligne
                            self.jouer_coup(type_ligne, i, j)
                
                # Mise à jour du slider et autres contrôles
                if event.type == pygame.MOUSEBUTTONUP or event.type == pygame.MOUSEMOTION:
                    self.slider_difficulte.gerer_evenement(event)
            
            # Tour de l'ordinateur
            if not self.jeu_termine and self.tour == 2:
                # Petit délai pour simuler la réflexion
                pygame.time.delay(300)
                self.coup_ordinateur()
            
            self.dessiner_grille()
            pygame.display.flip()
            self.horloge.tick(60)

def ecran_selection_taille():
    pygame.init()
    largeur_fenetre = 600
    hauteur_fenetre = 400
    fenetre = pygame.display.set_mode((largeur_fenetre, hauteur_fenetre))
    pygame.display.set_caption("Pipopinette - Sélection de la taille")
    
    # Créer des boutons pour les différentes tailles
    y_start = hauteur_fenetre // 3
    boutons = []
    tailles = list(TAILLES_GRILLE.keys())
    for i, nom_taille in enumerate(tailles):
        bouton = BoutonTaille(
            largeur_fenetre // 4,
            y_start + i * 70,
            largeur_fenetre // 2,
            50,
            f"{nom_taille} ({TAILLES_GRILLE[nom_taille]}x{TAILLES_GRILLE[nom_taille]})",
            TAILLES_GRILLE[nom_taille]
        )
        boutons.append(bouton)
    
    running = True
    horloge = pygame.time.Clock()
    
    # Titre et instructions
    font_titre = pygame.font.SysFont(None, 40)
    font_instructions = pygame.font.SysFont(None, 24)
    titre_surface = font_titre.render("Jeu de Pipopinette", True, NOIR)
    instructions_surface = font_instructions.render("Choisissez la taille de la grille", True, NOIR)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for bouton in boutons:
                    if bouton.verifier_clic(pos):
                        return bouton.taille
        
        # Surveiller les positions de la souris
        pos = pygame.mouse.get_pos()
        for bouton in boutons:
            bouton.verifier_survol(pos)
        
        # Dessiner l'écran
        fenetre.fill(FOND)
        
        # Afficher le titre et les instructions
        fenetre.blit(titre_surface, (largeur_fenetre // 2 - titre_surface.get_width() // 2, 30))
        fenetre.blit(instructions_surface, (largeur_fenetre // 2 - instructions_surface.get_width() // 2, 80))
        
        # Afficher les boutons
        for bouton in boutons:
            bouton.dessiner(fenetre)
        
        pygame.display.flip()
        horloge.tick(60)

def main():
    while True:
        # Sélection de la taille de la grille
        taille_grille = ecran_selection_taille()
        
        # Lancer le jeu avec la taille sélectionnée
        jeu = JeuPipopinette(taille_grille)
        resultat = jeu.executer()
        
        # Si le jeu renvoie "menu", on continue la boucle
        # Sinon (si le jeu a été fermé), on sort
        if resultat != "menu":
            break

if __name__ == "__main__":
    main()