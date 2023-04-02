# -*- coding: utf-8 -*-
from process import *

colors = [Orange, Eau, Bleu_clair, Corail, Cafe, Gris, Orange_clair, Orange_fonce, Violet, Bleu_pale, Bleu_fonce, Noir]

class Bouton():
    
    def __init__(self, x, y, l, h, coin = True):
        if coin:
            self.x, self.y = x, y
            self.centre = (x + l//2, y + h//2)
        else:
            self.x, self.y = x - l//2, y - h//2
            self.centre = x, y
        self.l, self.h = l, h
        
    def __contains__(self, clic):
        x, l, y, h = self.x, self.l, self.y, self.h
        return x <= clic[0] < x + l and y <= clic[1] < y + h

    def dessine(self, ecran, couleur = Blanc, alpha = 200):
        Affichage.dessine_rectangle(self.x, self.y, self.l, self.h, ecran,
                                    coin = True, couleur = couleur, alpha = alpha)
    
    def dessins(liste, ecran, couleur = Blanc, alpha = 200):
        for bouton in liste:
            Bouton.dessine(bouton, ecran, couleur = couleur, alpha = alpha)
        
    def ecrit_texte(self, texte, police, couleur, ecran, nom_police = None):
        Affichage.ecrit_texte(self.centre[0], self.centre[1], texte, police,
                              couleur, ecran, coin = False, nom_police = nom_police)
        
    def ecrit_textes(liste, textes, police, couleur, ecran, nom_police = None):
        n = len(liste)
        for i in range(n):
            bouton, texte = liste[i], textes[i]
            Bouton.ecrit_texte(bouton, texte, police, couleur, ecran,
                               nom_police = nom_police)
        
    def dessine_contour(self, couleur, epaisseur, ecran):
        Affichage.dessine_encadre(self.x, self.y, self.l, self.h, couleur,
                                  epaisseur, ecran)

    def dessine_icone(self, nom, ecran, colorkey = Blanc):
        image = pg.image.load(nom).convert_alpha()
        taille = (min(image.get_height(), self.h), min(image.get_width(), self.l))
        X, Y = ecran.get_size()
        taille = (X*taille[0]/1600, Y*taille[1]/900)
        if colorkey :
            image.set_colorkey(colorkey)
        rect = pg.Rect(0, 0, taille[0], taille[1])
        image = pg.transform.scale(image, taille)
        rect.center = (X*self.centre[0]/1600, X*self.centre[1]/1600)
        ecran.blit(image, rect)
    
    def dessine_icones(liste, noms, ecran, colorkey = Blanc):
        n = len(liste)
        for i in range(n):
            bouton, nom = liste[i], noms[i]
            Bouton.dessine_icone(bouton, nom, ecran, colorkey = colorkey)
    
    def fill(self, couleur, ecran):
        Bouton.dessine(self, ecran, couleur = couleur, alpha = 255)
    
    def distance(clic, self):
        return np.sqrt((clic[0]-self.centre[0])**2 + (clic[1]-self.centre[1])**2)

# +
class Affichage():
            
    def dessine_image(x, y, nom, ecran, coin = True, taille = None, colorkey = None,
                     intensite = 255, angle = 0):
        X, Y = ecran.get_size()
        x, y = X*x/1600, Y*y/900
        if nom != None:
            image = pg.image.load(nom).convert_alpha()
            if colorkey or intensite != 255:
                image.set_alpha(intensite)
                if colorkey == None:
                    colorkey = Blanc
                image.set_colorkey(colorkey)
            if taille:
                taille = (X*taille[0]/1600, Y*taille[1]/900)
                rect = pg.Rect(x, y, taille[0], taille[1])
                image = pg.transform.scale(image, taille)
            else:
                rect = pg.Rect(x, y, image.get_height(), image.get_width())
            image = pg.transform.rotozoom(image, angle, 1)
            if not coin :
                rect.center = (x, y)
            ecran.blit(image, rect)
            
    def dessine_matrice(x, y, matrice, ecran, coin = True, taille = None, dec = False, cmap = None):
        if dec:
            matrice[matrice >= 1] = 1
            matrice[matrice <= 0] = 0
            matrice = (255*matrice).astype(np.uint8)
        if len(matrice.shape) == 2:
            matrice = np.repeat(matrice[:, :, np.newaxis], 3, axis=2)
            if cmap:
                matrice = plt.imshow(matrice, cmap = 'viridis')
        X, Y = ecran.get_size()
        l, h = matrice.shape[0], matrice.shape[1]
        surface = pg.Surface((l, h))
        pg.surfarray.blit_array(surface, matrice)
        x, y = X*x/1600, Y*y/900
        if taille:
            l, h = taille[0], taille[1]
        l, h = X*l/1600, Y*h/900
        surface = pg.transform.scale(surface, (l, h))
        if coin:
            rect = pg.Rect(x, y, l, h)
        else:
            rect = pg.Rect(x - l//2, y - h//2, l, h)
        ecran.blit(surface, rect)

    def dessine_motif(x, y, motif, couleur, ecran):
        X, Y = ecran.get_size()
        x, y = X*x/1600, Y*y/900
        for pixel in motif:
            rect = pg.Rect(x + pixel[0], y + pixel[1], 1, 1)
            pg.draw.rect(ecran, couleur, rect)
    
    def dessine_rectangle(x, y, l, h, ecran, coin = False, couleur = Blanc, alpha = 140):
        X, Y = ecran.get_size()
        x, y = X*x/1600, Y*y/900
        l, h = X*l/1600, Y*h/900
        s = pg.Surface((l, h))
        s.set_alpha(alpha)
        s.fill(couleur)
        if not(coin): #x et y sont les coordonnées du centre
            ecran.blit(s, (x - l//2, y - h//2))
        else:
            ecran.blit(s, (x, y))
            
    def dessine_encadre(x, y, l, h, couleur, epaisseur, ecran, coin = True):
        X, Y = ecran.get_size()
        x, y = X*x/1600, Y*y/900
        l, h = np.ceil(X*l/1600), np.ceil(Y*h/900)
        epaisseur = np.ceil(X*epaisseur/1600)
        if coin:
            rects = [pg.Rect(x - epaisseur, y - epaisseur, epaisseur, h + epaisseur),
                     pg.Rect(x, y - epaisseur, l + epaisseur, epaisseur),
                     pg.Rect(x + l, y, epaisseur, h + epaisseur),
                     pg.Rect(x - epaisseur, y + h, l + epaisseur, epaisseur)]
        else:
            rects = [pg.Rect(x - epaisseur - l//2, y - epaisseur - h//2, epaisseur, h + epaisseur),
                     pg.Rect(x - l//2, y - epaisseur - h//2, l + epaisseur, epaisseur),
                     pg.Rect(x + l//2, y - h//2, epaisseur, h + epaisseur),
                     pg.Rect(x - epaisseur - l//2, y + h//2, l + epaisseur, epaisseur)]
        for rect in rects:
            pg.draw.rect(ecran, couleur, rect)
    
    def ecrit_texte(x, y, texte, police, couleur, ecran, coin = False,
                    nom_police = None, angle = 0):
        X, Y = ecran.get_size()
        x, y = X*x/1600, Y*y/900
        police = int(X*police/1600)
        police = pg.font.Font(nom_police, police)
        texte = police.render(texte, True, couleur)
        rect = texte.get_rect()
        if not(coin):
            rect.center = (x, y)
        else:
            rect.topleft = (x, y)
        texte = pg.transform.rotozoom(texte, angle, 1)
        ecran.blit(texte, rect)
        
    def ecrit_textes(x, y, textes, marges, polices, couleur, ecran, coin = False,
                    nom_police = None, angle = 0):
        if len(textes) > 0:
            X, Y = ecran.get_size()
            Affichage.ecrit_texte(x, y, textes[0], polices[0], couleur, ecran, coin = coin,
                                  nom_police = nom_police, angle = angle)
            for i in range(len(textes)-1):
                y += marges[i]*X/1600
                Affichage.ecrit_texte(x, y, textes[i+1], polices[i+1], couleur, ecran,
                                      coin = coin, nom_police = nom_police, angle = angle)
        
    def sous_ecran(vrai_ecran):
        X, Y = vrai_ecran.get_size()
        if X/Y >= 16/9:
            X2, Y2 = 16*Y/9, Y
        else:
            X2, Y2 = X, 9*X/16
        return pg.Surface((X2, Y2))
    
    def max_blit(vrai_ecran, ecran):
        X, Y = vrai_ecran.get_size()
        if X/Y >= 16/9:
            X2, Y2 = 16*Y/9, Y
            ox, oy = (X-X2)/2, 0
        else:
            X2, Y2 = X, 9*X/16
            ox, oy = 0, (Y-Y2)/2
        rect = pg.Rect(ox, oy, X2, Y2)
        vrai_ecran.blit(ecran, rect)
        pg.display.update()
        
    def convertisseur(clic, vrai_ecran, ecran):
        X, Y = vrai_ecran.get_size()
        if X/Y >= 16/9:
            X2, Y2 = 16*Y/9, Y
            ox, oy = (X-X2)/2, 0
        else:
            X2, Y2 = X, 9*X/16
            ox, oy = 0, (Y-Y2)/2
        x, y = clic
        return ((x-ox)*1600/X2, (y-oy)*900/Y2)
    
    def resize(image, size = (700, 700)):
        if len(image.shape) == 2:
            return cv2.resize(image, size, interpolation = cv2.INTER_CUBIC)
        else:
            r, g, b = image[: , : , 0], image[: , : , 1], image[: , : , 2]
            r = cv2.resize(r, size, interpolation = cv2.INTER_CUBIC)
            g = cv2.resize(g, size, interpolation = cv2.INTER_CUBIC)
            b = cv2.resize(b, size, interpolation = cv2.INTER_CUBIC)
            return np.array([r, g, b]).transpose([1, 2, 0])
        
    def bilan_global(n_val = 10):
        names = np.array(glob("data/val/labels/*.npy"))[: n_val]
        n_val = len(names)
        network_names = os.listdir("networks")
        network_names.remove('plots')
        network_names.append('plots')
        n_networks = len(network_names)
        networks = []
        for network_name in network_names:
            if network_name != 'plots':
                networks.append(Deconv.load(network_name))
        points = np.zeros((n_networks, 2, n_val))
        psf = np.loadtxt("data/psf/psf_pan_2d_30cm_from10cm.txt")
        psf = psf / psf.sum() # normalizing the psf
        for i in range(n_val):
            print(i, end='\r')
            id_name = names[i][len(f"data/val/labels/"): -4]
            label = np.load("data/val/labels/" + id_name + ".npy")[0]
            input_ = np.load("data/val/images/" + id_name + ".npy")[0]
            input2 = np.load("data/val/RLimages/" + id_name + ".npy")[0]
            outputs = []
            for j in range(n_networks-1):
                network = networks[j]
                if network.history['RL']:
                    output_tensor = network(torch.from_numpy(input2).float()[None, None, ...])
                else:
                    output_tensor = network(torch.from_numpy(input_).float()[None, None, ...])
                outputs.append(output_tensor.detach().numpy()[0][0])
            outputs.append(restoration.wiener(input_, psf, 1))
            denominateur = len(label)*np.mean(label)
            rmses = np.zeros(n_networks)
            super_resos = np.zeros(n_networks)
            for j in range(n_networks):
                rmses[j] = np.sqrt(np.sum((label - outputs[j])**2))/denominateur
                rapport = Processing.cross_PSD(label, outputs[j])/np.sqrt(Processing.PSD(label)*Processing.PSD(outputs[j])+0.01)
                nperseg = 100
                rapport[nperseg//3: 2*nperseg//3, nperseg//3: 2*nperseg//3] = 0
                super_resos[j] = 9*np.mean(rapport)/8
            titres = ["Output (" + network + ")" for network in network_names]
            points[: , 0, i] = 100*rmses
            points[: , 1, i] = 100*super_resos
        plt.figure()
        scatters = []
        for j in range(n_networks):
            scatters.append(plt.scatter(points[j, 0], points[j, 1], marker='o', color=np.random.rand(3,)))
        plt.xlabel("n-RMSE (%)")
        plt.ylabel("Super-resolution (%)")
        plt.legend(tuple(scatters), tuple(titres), loc='best', ncol=2, fontsize=10) #scatterpoints=1
        plt.savefig("networks/plots/scatter.jpg")
        plt.close()
        scatter = np.transpose(io.imread("networks/plots/scatter.jpg"), [1, 0, 2])
        
    def rmse_super_reso(label, input_, outputs):
        n = len(outputs)
        rmses = np.zeros((n, n))
        super_resos = np.zeros((n, n))
        denominateur = len(label)*np.mean(label)
        for i in range(n):
            for j in range(n):
                rmses[i, j] = np.sqrt(np.sum((label - outputs[i][j])**2))/denominateur
                rapport = Processing.cross_PSD(label, outputs[i][j])/np.sqrt(Processing.PSD(label)*Processing.PSD(outputs[i][j])+0.01)
                nperseg = 100
                rapport[nperseg//3: 2*nperseg//3, nperseg//3: 2*nperseg//3] = 0
                super_resos[i][j] = 100*9*np.mean(rapport)/8
        plt.figure()
        plt.imshow(rmses)
        plt.xlabel("Power of cross-entropy coeff")
        plt.ylabel("Power of Fourier angle coeff")
        plt.colorbar()
        plt.savefig("networks/plots/rmses.jpg")
        plt.close()
        plt.figure()
        plt.imshow(super_resos)
        plt.xlabel("Power of cross-entropy coeff")
        plt.ylabel("Power of Fourier angle coeff")
        plt.colorbar()
        plt.savefig("networks/plots/super_resos.jpg")
        plt.close()
        rmses = np.transpose(io.imread("networks/plots/rmses.jpg"), [1, 0, 2])
        super_resos = np.transpose(io.imread("networks/plots/super_resos.jpg"), [1, 0, 2])
        return rmses, super_resos
    
#Affichage.bilan_global(n_val = 10)


# -

def menu():
    pg.init()
    vrai_ecran = pg.display.set_mode((0,0), pg.FULLSCREEN)
    vrai_ecran.fill(Noir)
    X, Y = 1600, 900
    ecran = Affichage.sous_ecran(vrai_ecran)
    bouton_croix = Bouton(1540, 10, 50, 50)
    bouton_titre = Bouton(800, 50, 200, 60, coin = False)
    bouton_train = Bouton(100, 100, 200, 40)
    bouton_val = Bouton(100, 150, 200, 40)
    bouton_test = Bouton(100, 200, 200, 40)
    bouton_test1 = Bouton(1250, 760, 300, 60)
    bouton_test2 = Bouton(1250, 830, 300, 60)
    if not os.path.exists("examples/"):
        os.mkdir("examples/")
    if not os.path.exists("networks/plots/"):
        os.mkdir("networks/plots/")
    networks = os.listdir("networks")
    networks.remove('plots')
    networks.append('plots')
    n_networks = len(networks)
    boutons_networks, boutons_croix = [], []
    for i in range(n_networks-1):
        boutons_networks.append(Bouton(50, 300 + i*60, 300, 50))
    labels = [np.array(glob("data/train/labels/*.npy")),
              np.array(glob("data/val/labels/*.npy")),
              np.array(glob("data/test/labels/*.npy"))]
    nombre_cote = int(4000/len(np.load(labels[0][0])[0]))
    cartes = list(glob("data/satellite_images/*.tif"))
    n_cartes = len(cartes)
    train_ids, val_ids, test_ids = {}, {}, {}
    for i in range(n_cartes):
        train_ids[int(cartes[i][len(f"data/satellite_images/"): -4])] = set()
        val_ids[int(cartes[i][len(f"data/satellite_images/"): -4])] = set()
        test_ids[int(cartes[i][len(f"data/satellite_images/"): -4])] = set()
    for label in labels[0]:
        id_ = label[len(f"data/train/images/"): -4]
        id_ = id_.split(".")
        train_ids[int(id_[0])].add((int(id_[1]), int(id_[2])))
    for label in labels[1]:
        id_ = label[len(f"data/val/images/"): -4]
        id_ = id_.split(".")
        val_ids[int(id_[0])].add((int(id_[1]), int(id_[2])))
    for label in labels[2]:
        id_ = label[len(f"data/test/images/"): -4]
        id_ = id_.split(".")
        test_ids[int(id_[0])].add((int(id_[1]), int(id_[2])))
    ind_carte, n_cartes = 0, len(cartes)
    num_carte = int(cartes[ind_carte][len(f"data/satellite_images/"): -4])
    carte = cv2.imread(cartes[ind_carte]).mean(axis=2)
    carte = cv2.resize(carte, (800, 800), interpolation = cv2.INTER_CUBIC)
    running = True
    clock = pg.time.Clock()
    while running:
        clock.tick(15)
        position = Affichage.convertisseur(pg.mouse.get_pos(), vrai_ecran, ecran)
        i, j, clic = -1, -1, None
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
            if event.type == pg.MOUSEBUTTONDOWN:
                clic = Affichage.convertisseur(pg.mouse.get_pos(), vrai_ecran, ecran)
        ecran.fill(Noir)
        bouton_train.dessine(ecran)
        bouton_val.dessine(ecran)
        bouton_test.dessine(ecran)
        bouton_train.ecrit_texte("Train", 40, Eau, ecran)
        bouton_val.ecrit_texte("Validation", 40, Orange, ecran)
        bouton_test.ecrit_texte("Test", 40, Or, ecran)
        bouton_test1.dessine(ecran)
        bouton_test1.ecrit_texte("Test 1", 40, Noir, ecran)
        bouton_test2.dessine(ecran)
        bouton_test2.ecrit_texte("Test 2", 40, Noir, ecran)
        for k in range(n_networks-1):
            boutons_networks[k].dessine(ecran)
            boutons_networks[k].ecrit_texte(networks[k], 50, Noir, ecran)
        Affichage.dessine_matrice(800, 450, carte, ecran, coin = False)
        i, j = int(np.floor(nombre_cote*(position[0]-400)/800)), int(np.floor(nombre_cote*(position[1]-50)/800))
        if 0 <= i <= nombre_cote and 0 <= j <= nombre_cote:
            cote = int(800/nombre_cote)
            if (i, j) in train_ids[num_carte]:
                Affichage.dessine_encadre(400 + i*cote, 50 + j*cote, cote, cote, Eau, 2, ecran)
            elif (i, j) in val_ids[num_carte]:
                Affichage.dessine_encadre(400 + i*cote, 50 + j*cote, cote, cote, Orange, 2, ecran)
            elif (i, j) in test_ids[num_carte]:
                Affichage.dessine_encadre(400 + i*cote, 50 + j*cote, cote, cote, Or, 2, ecran)
        bouton_croix.dessine(ecran)
        bouton_croix.dessine_icone("Croix.png", ecran)
        if position in bouton_croix:
            bouton_croix.dessine_contour(Or, 4, ecran)
        elif position in bouton_test1:
            bouton_test1.dessine_contour(Or, 4, ecran)
        elif position in bouton_test2:
            bouton_test2.dessine_contour(Or, 4, ecran)
        if clic:
            x, y = clic
            if clic in bouton_croix:
                running = False
            elif clic in bouton_test1:
                test1(vrai_ecran, new = False)
            elif clic in bouton_test2:
                test2(vrai_ecran, ajout = ("", 11))
            elif clic and 0 <= i <= nombre_cote and 0 <= j <= nombre_cote:
                if (i, j) in train_ids[num_carte]:
                    id_ = f"{num_carte}.{i}.{j}"
                    jeu(id_, "train", vrai_ecran)
                if (i, j) in val_ids[num_carte]:
                    id_ = f"{num_carte}.{i}.{j}"
                    jeu(id_, "val", vrai_ecran)
                if (i, j) in test_ids[num_carte]:
                    id_ = f"{num_carte}.{i}.{j}"
                    jeu(id_, "test", vrai_ecran)
            elif clic and x < 350:
                ind_carte = (ind_carte - 1)%n_cartes
                num_carte = int(cartes[ind_carte][len(f"data/satellite_images/"): -4])
            elif clic and x > 1250:
                ind_carte = (ind_carte + 1)%n_cartes
                num_carte = int(cartes[ind_carte][len(f"data/satellite_images/"): -4])
            carte = cv2.imread(cartes[ind_carte]).mean(axis=2)
            carte = cv2.resize(carte, (800, 800), interpolation = cv2.INTER_CUBIC)
        Affichage.max_blit(vrai_ecran, ecran)
    pg.quit()

def test1(vrai_ecran, new = False):
    score_test1 = None
    val_name = rd.choice(glob("data/val/labels/*.npy"))
    id_name = val_name[len(f"data/val/labels/"): -4]
    label = np.load("data/val/labels/" + id_name + ".npy")[0]
    input_ = np.load("data/val/images/" + id_name + ".npy")[0]
    input2 = np.load("data/val/RLimages/" + id_name + ".npy")[0]
    networks = os.listdir("networks")
    networks.remove('plots')
    networks.append('plots')
    n_networks = len(networks)
    outputs = []
    for name in networks:
        if name != 'plots':
            network = Deconv.load(name)
            if network.history['RL']:
                output_tensor = network(torch.from_numpy(input2).float()[None, None, ...])
            else:
                output_tensor = network(torch.from_numpy(input_).float()[None, None, ...])
            outputs.append(output_tensor.detach().numpy()[0][0])
    psf = np.loadtxt("data/psf/psf_pan_2d_30cm_from10cm.txt")
    psf = psf / psf.sum() # normalizing the psf
    outputs.append(restoration.wiener(input_, psf, 1))
    label = Affichage.resize(label)
    for i in range(n_networks):
        outputs[i] = Affichage.resize(outputs[i])
    ind = 0
    image = outputs[ind]
    vrai_ecran.fill(Noir)
    X, Y = 1600, 900
    ecran = Affichage.sous_ecran(vrai_ecran)
    bouton_croix = Bouton(1540, 10, 50, 50)
    bouton_consigne = Bouton(800, 50, 500, 60, coin = False)
    bouton_choix = Bouton(430, 850, 200, 60, coin = False)
    running = True
    clock = pg.time.Clock()
    while running:
        clock.tick(10)
        position = Affichage.convertisseur(pg.mouse.get_pos(), vrai_ecran, ecran)
        clic = None
        maj_zoom = False
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                if event.key == pg.K_SPACE or event.key == pg.K_RETURN:
                    flip = not(flip)
            if event.type == pg.MOUSEBUTTONDOWN:
                clic = Affichage.convertisseur(pg.mouse.get_pos(), vrai_ecran, ecran)
        ecran.fill(Noir)
        Affichage.dessine_matrice(430, 450, label, ecran, coin = False, dec = True)
        Affichage.dessine_matrice(1170, 450, image, ecran, coin = False, dec = True)
        couleurs_carres = n_networks*[Gris]
        couleurs_carres[ind] = Or
        for k in range(n_networks):
            Affichage.dessine_rectangle(1170 - (n_networks//2 - k)*50, 850, 40, 40, ecran, coin = False,
                                        couleur = couleurs_carres[k], alpha = 255)
        bouton_croix.dessine(ecran)
        bouton_croix.dessine_icone("Croix.png", ecran)
        bouton_consigne.dessine(ecran)
        bouton_consigne.ecrit_texte("Please choose the best image", 30, Noir, ecran)
        bouton_choix.dessine(ecran)
        bouton_choix.ecrit_texte("Choose it !", 30, Noir, ecran)
        if position in bouton_croix:
            bouton_croix.dessine_contour(Or, 4, ecran)
        elif position in bouton_choix:
            bouton_choix.dessine_contour(Or, 4, ecran)
        if clic:
            x, y = clic
            if clic in bouton_croix:
                running = False
            elif clic in bouton_choix:
                if not os.path.exists("data/eval/"):
                    os.mkdir("data/eval/")
                if score_test1 == None:
                    if os.path.exists("data/eval/score_test1.yaml"):
                        with open("data/eval/score_test1.yaml", 'r') as fichier:
                            score_test1 = yaml.load(fichier, Loader = yaml.UnsafeLoader)
                    else:
                        score_test1 = {id_name: []}
                if id_name in score_test1:
                    score_test1[id_name].append(networks[ind])
                else:
                    score_test1[id_name] = [networks[ind]]
                with open("data/eval/score_test1.yaml", "w") as fichier:
                    fichier.write(yaml.dump(score_test1, Dumper = yaml.Dumper))
                val_name = rd.choice(glob("data/val/labels/*.npy"))
                id_name = val_name[len(f"data/val/labels/"): -4]
                label = np.load("data/val/labels/" + id_name + ".npy")[0]
                input_ = np.load("data/val/images/" + id_name + ".npy")[0]
                input2 = np.load("data/val/RLimages/" + id_name + ".npy")[0]
                networks = os.listdir("networks")
                networks.remove('plots')
                networks.append('plots')
                n_networks = len(networks)
                outputs = []
                for name in networks:
                    if name != 'plots':
                        network = Deconv.load(name)
                        if network.history['RL']:
                            output_tensor = network(torch.from_numpy(input2).float()[None, None, ...])
                        else:
                            output_tensor = network(torch.from_numpy(input_).float()[None, None, ...])
                        outputs.append(output_tensor.detach().numpy()[0][0])
                psf = np.loadtxt("data/psf/psf_pan_2d_30cm_from10cm.txt")
                psf = psf / psf.sum() # normalizing the psf
                outputs.append(restoration.wiener(input_, psf, 1))
                label = Affichage.resize(label)
                for i in range(n_networks):
                    outputs[i] = Affichage.resize(outputs[i])
                ind = 0
                image = outputs[ind]
            elif x <= 1170:
                ind = (ind-1)%n_networks
            elif x > 1170:
                ind = (ind+1)%n_networks
            image = outputs[ind]
        Affichage.max_blit(vrai_ecran, ecran)


def test2(vrai_ecran, ajout = ("", 11)):
    val_name = rd.choice(glob("data/val/labels/*.npy"))
    id_name = val_name[len(f"data/val/labels/"): -4]
    label = np.load("data/val/labels/" + id_name + ".npy")[0]
    input_ = np.load("data/val/images/" + id_name + ".npy")[0]
    input2 = np.load("data/val/RLimages/" + id_name + ".npy")[0]
    input3 = np.load("data/val/mix_images/" + id_name + ".npy")[0]
    networks = []
    n = ajout[1]
    for i in range(n):
        networks.append([])
        for j in range(n):
            with open("grid" + ajout[0] + f"/{i-n//2}_{j-n//2}_cnn" + "/history.yaml", 'r') as fichier:
                history = yaml.load(fichier, Loader = yaml.UnsafeLoader)
                network = Deconv.load("grid" + ajout[0] + f"/{i-n//2}_{j-n//2}_cnn", branch = False)
            networks[i].append(network)
    outputs = []
    for i in range(n):
        outputs.append([])
        for j in range(n):
            network = networks[i][j]
            if network.history['RL']:
                output_tensor = network(torch.from_numpy(input2).float()[None, None, ...])
            elif network.history['mix']:
                output_tensor = network(torch.from_numpy(input3).float()[None, None, ...])
            else:
                output_tensor = network(torch.from_numpy(input_).float()[None, None, ...])
            output = output_tensor.detach().numpy()[0][0]
            output = Affichage.resize(output)
            outputs[i].append(output)
    i, j = n//2, n//2
    vrai_ecran.fill(Noir)
    X, Y = 1600, 900
    ecran = Affichage.sous_ecran(vrai_ecran)
    bouton_croix = Bouton(1540, 10, 50, 50)
    bouton_gauche = Bouton(300, 150, 100, 600)
    bouton_bas = Bouton(500, 800, 600, 100)
    bouton_i = Bouton(350, 450, 20, 20, coin = False)
    bouton_j = Bouton(800, 850, 20, 20, coin = False)
    bouton_valide = Bouton(1200, 430, 300, 40)
    #rmses, super_resos = rmse_super_reso(label, input_, outputs)
    running = True
    clock = pg.time.Clock()
    while running:
        clock.tick(10)
        position = Affichage.convertisseur(pg.mouse.get_pos(), vrai_ecran, ecran)
        clic = None
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
            if event.type == pg.MOUSEBUTTONDOWN:
                clic = Affichage.convertisseur(pg.mouse.get_pos(), vrai_ecran, ecran)
        ecran.fill(Noir)
        Affichage.dessine_matrice(800, 450, outputs[i][j], ecran, coin = False, dec = True)
        Affichage.dessine_rectangle(350, 450, 10, 600, ecran, couleur = Gris, alpha = 255)
        Affichage.dessine_rectangle(800, 850, 600, 10, ecran, couleur = Gris, alpha = 255)
        bouton_croix.dessine(ecran)
        bouton_croix.dessine_icone("Croix.png", ecran)
        bouton_valide.dessine(ecran)
        bouton_valide.ecrit_texte("Choose it as the best cnn", 30, Noir, ecran)
        bouton_i.dessine(ecran, couleur = Eau, alpha = 255)
        bouton_j.dessine(ecran, couleur = Eau, alpha = 255)
        if position in bouton_croix:
            bouton_croix.dessine_contour(Or, 4, ecran)
        elif position in bouton_valide:
            bouton_valide.dessine_contour(Or, 4, ecran)
        if clic:
            x, y = clic
            if clic in bouton_croix:
                running = False
            elif clic in bouton_valide:
                running = False
                networks[i][j].name = "best_cnn"
                networks[i][j].save()
        if pg.mouse.get_pressed()[0]:
            px, py = position
            if position in bouton_gauche:
                i = int(max(0, min(n-1, n*(py - bouton_gauche.y)/bouton_gauche.h)))
                bouton_i.y = max(150, min(py, 750))
            elif position in bouton_bas:
                j = int(max(0, min(n-1, n*(px - bouton_bas.x)/bouton_bas.l)))
                bouton_j.x = max(500, min(px, 1100))
        Affichage.max_blit(vrai_ecran, ecran)


def jeu(id_name, folder, vrai_ecran):
    print(id_name)
    label = np.load("data/" + folder + "/labels/" + id_name + ".npy")[0]
    #low_label = cv2.resize(label[1: : 3, 1: : 3], label.shape, interpolation = cv2.INTER_CUBIC)
    input_ = np.load("data/" + folder + "/images/" + id_name + ".npy")[0]
    input2 = np.load("data/" + folder + "/RLimages/" + id_name + ".npy")[0]
    input3 = np.load("data/" + folder + "/mix_images/" + id_name + ".npy")[0]
    io.imsave("examples/label.jpg", label)
    io.imsave("examples/input.jpg", input_)
    io.imsave("examples/RLinput.jpg", input2)
    io.imsave("examples/mix_input.jpg", input3)
    networks = os.listdir("networks")
    networks.remove('plots')
    networks.append('wiener')
    n_networks = len(networks)
    outputs = []
    for name in networks:
        if name != 'wiener':
            network = Deconv.load(name)
            if network.history['RL']:
                output_tensor = network(torch.from_numpy(input2).float()[None, None, ...])
            elif network.history['mix']:
                output_tensor = network(torch.from_numpy(input3).float()[None, None, ...])
            else:
                output_tensor = network(torch.from_numpy(input_).float()[None, None, ...])
            output = output_tensor.detach().numpy()[0][0]
            print(np.mean(label), np.mean(output))
            io.imsave("examples/" + name + ".jpg", output)
            outputs.append(output)
    psf = np.loadtxt("data/psf/psf_pan_2d_30cm_from10cm.txt")
    psf = psf / psf.sum() # normalizing the psf
    outputs.append(restoration.wiener(input_, psf, 1))
    denominateur = len(label)*np.mean(label)
    rmses = [np.sqrt(np.sum((label - input_)**2))/denominateur,
            np.sqrt(np.sum((label - input2)**2))/denominateur,
            np.sqrt(np.sum((label - input3)**2))/denominateur,]
    super_resos = []
    psds = []
    super_reso, hyper_reso = Processing.save_cross_psd('plots', label, input_, nperseg = 100,
                                               noverlap = 10, super_reso = True, hyper_reso = True)
    super_resos.append(super_reso)
    psds.append(np.transpose(io.imread("networks/plots/psd.jpg"), [1, 0, 2]))
    io.imsave("examples/input_psd.jpg", io.imread("networks/plots/psd.jpg"))
    super_reso, hyper_reso = Processing.save_cross_psd('plots', label, input2, nperseg = 100,
                                               noverlap = 10, super_reso = True, hyper_reso = True)
    super_resos.append(super_reso)
    psds.append(np.transpose(io.imread("networks/plots/psd.jpg"), [1, 0, 2]))
    io.imsave("examples/RLinput_psd.jpg", io.imread("networks/plots/psd.jpg"))
    super_reso, hyper_reso = Processing.save_cross_psd('plots', label, input3, nperseg = 100,
                                               noverlap = 10, super_reso = True, hyper_reso = True)
    super_resos.append(super_reso)
    psds.append(np.transpose(io.imread("networks/plots/psd.jpg"), [1, 0, 2]))
    io.imsave("examples/mix_input_psd.jpg", io.imread("networks/plots/psd.jpg"))
    textes = [f"RMSE label - input : {round(100*rmses[0], 2)} %",
              f"Super-resolution (input) : {round(100*super_resos[0], 2)} %",
              f"RMSE label - RL input : {round(100*rmses[1], 2)} %",
             f"Super-resolution (RL input) : {round(100*super_resos[1], 2)} %",
             f"RMSE label - mix input : {round(100*rmses[2], 2)} %",
             f"Super-resolution (mix input) : {round(100*super_resos[2], 2)} %",]
    for i in range(n_networks-1):
        rmses.append(np.sqrt(np.sum((label - outputs[i])**2))/denominateur)
        super_reso, hyper_reso = Processing.save_cross_psd(networks[i], label, outputs[i], nperseg = 100,
                                               noverlap = 10, super_reso = True, hyper_reso = True)
        super_resos.append(super_reso)
        psds.append(np.transpose(io.imread("networks/" + networks[i] + "/psd.jpg"), [1, 0, 2]))
        textes += [f"RMSE label-output ({networks[i]}) : {round(100*rmses[-1], 2)} %",
                   f"Super-resolution ({networks[i]}) : {round(100*super_resos[-1], 2)} %"]
    rmses.append(np.sqrt(np.sum((label - outputs[-1])**2))/denominateur)
    textes += [f"RMSE label-output (wiener) : {round(100*rmses[-1], 2)} %"]
    super_reso, hyper_reso = Processing.save_cross_psd("plots", label, outputs[-1], nperseg = 100, noverlap = 10,
                                           super_reso = True, hyper_reso = True)
    super_resos.append(super_reso)
    psds.append(np.transpose(io.imread("networks/plots/psd.jpg"), [1, 0, 2]))
    titres = ["Label (10 cm)", "Input", "RL input", "Mix input"]
    titres += ["Output (" + network + ")" for network in networks]
    titres += ["Cross PSD (input)", "Cross PSD (RL input)", "Cross PSD (mix input)"]
    titres += ["Cross PSD (" + network + ")" for network in networks] + ["Overview"]
    plt.figure()
    scatters = []
    for i in range(n_networks + 3): #three inputs and all networks (including wiener)
        color = (colors[i][0]/255, colors[i][1]/255, colors[i][2]/255)
        scatters.append(plt.scatter(100*rmses[i], 100*super_resos[i], marker='o', color=color))
    plt.xlabel("n-RMSE (%)")
    plt.ylabel("Super-resolution (%)")
    plt.legend(tuple(scatters), tuple(titres[1: 4 + n_networks]), scatterpoints=1, loc='best', ncol=2, fontsize=10)
    plt.savefig("scatter.jpg")
    plt.close()
    scatter = np.transpose(io.imread("scatter.jpg"), [1, 0, 2])
    matrices = [label, input_, input2, input3] + outputs + psds + [scatter]
    n_matrices = 2*n_networks + 8 #label, input, inputRL, inputmix, psdinput, psdRL, psdmix, scatter
    for i in range(n_matrices):
        matrices[i] = Affichage.resize(matrices[i])
    ind = 0
    description = []
    compared = [None, None]
    image = matrices[ind]
    vrai_ecran.fill(Noir)
    X, Y = 1600, 900
    ecran = Affichage.sous_ecran(vrai_ecran)
    bouton_croix = Bouton(1540, 10, 50, 50)
    bouton_compare1 = Bouton(1225, 700, 300, 40)
    bouton_compare2 = Bouton(1225, 750, 300, 40)
    bouton_compare = Bouton(1225, 810, 300, 60)
    bouton_titre = Bouton(800, 50, 500, 60, coin = False)
    sample_size = len(image)
    running = True
    clock = pg.time.Clock()
    while running:
        clock.tick(10)
        position = Affichage.convertisseur(pg.mouse.get_pos(), vrai_ecran, ecran)
        clic = None
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
            if event.type == pg.MOUSEBUTTONDOWN:
                clic = Affichage.convertisseur(pg.mouse.get_pos(), vrai_ecran, ecran)
        ecran.fill(Noir)
        Affichage.dessine_matrice(800, 450, image, ecran, coin = False, dec = (ind < 4 + n_networks))
        bouton_croix.dessine(ecran)
        bouton_croix.dessine_icone("Croix.png", ecran)
        bouton_compare1.dessine(ecran)
        bouton_compare2.dessine(ecran)
        if compared[0] == None:
            bouton_compare1.ecrit_texte("Add to comparator", 40, Noir, ecran)
        else:
            bouton_compare1.ecrit_texte(titres[compared[0]], 40, Noir, ecran)
        if compared[1] == None:
            bouton_compare2.ecrit_texte("Add to comparator", 40, Noir, ecran)
        else:
            bouton_compare2.ecrit_texte(titres[compared[1]], 40, Noir, ecran)
        if compared[0] != None and compared[1] != None:
            bouton_compare.dessine(ecran)
            bouton_compare.ecrit_texte("Start comparaison", 40, Noir, ecran)
        bouton_titre.dessine(ecran)
        bouton_titre.ecrit_texte(titres[ind], 50, Noir, ecran)
        Affichage.ecrit_textes(250, 450 - len(description)*20, description, (len(description)-1)*[40],
                               len(description)*[28], Blanc, ecran)
        Affichage.ecrit_textes(1375, 450 - len(textes)*20, textes, (len(textes)-1)*[40], len(textes)*[28], Blanc, ecran)
        couleurs_carres = n_matrices*[Gris]
        couleurs_carres[ind] = Or
        for k in range(n_matrices):
            Affichage.dessine_rectangle(800 - (n_matrices//2 - k)*50, 850, 40, 40, ecran, coin = False,
                                        couleur = couleurs_carres[k], alpha = 255)
        if position in bouton_croix:
            bouton_croix.dessine_contour(Or, 4, ecran)
        elif position in bouton_compare1:
            bouton_compare1.dessine_contour(Or, 4, ecran)
        elif position in bouton_compare2:
            bouton_compare2.dessine_contour(Or, 4, ecran)
        elif compared[0] != None and compared[1] != None and position in bouton_compare:
            bouton_compare.dessine_contour(Or, 4, ecran)
        if clic:
            x, y = clic
            if clic in bouton_croix:
                running = False
            elif clic in bouton_compare1:
                if compared[0] == None:
                    compared[0] = ind
                else:
                    compared[0] = None
            elif clic in bouton_compare2:
                if compared[1] == None:
                    compared[1] = ind
                else:
                    compared[1] = None
            elif compared[0] != None and compared[1] != None and clic in bouton_compare:
                comparaison(compared, matrices, titres, vrai_ecran)
            elif x <= X//2:
                ind = (ind-1)%n_matrices
            elif x > X//2:
                ind = (ind+1)%n_matrices
            image = matrices[ind]
            description = []
            if ind >= 4:
                name = networks[(ind-4)%n_networks]
                if name != 'wiener': #cas de wiener exclu
                    with open("networks/" + name + "/history.yaml", 'r') as fichier:
                        history = yaml.load(fichier, Loader = yaml.UnsafeLoader)
                    description = ["Type : Deconv",
                                   "L1 norme Fourier : " + str(history["norm_coeff"]),
                                   "L1 angle Fourier : " + str(history["angle_coeff"]),
                                   "L2 image : " + str(history["energy_coeff"]),
                                   "Total variation : " + str(history["var_coeff"]),
                                   "L1 image : " + str(history["abs_coeff"]),
                                   "Cross correlation : " + str(history["cross_coeff"])]    
        Affichage.max_blit(vrai_ecran, ecran)


def comparaison(compared, matrices, titres, vrai_ecran):
    image1, image2 = matrices[compared[0]], matrices[compared[1]]
    titre1, titre2 = titres[compared[0]], titres[compared[1]]
    vrai_ecran.fill(Noir)
    X, Y = 1600, 900
    ecran = Affichage.sous_ecran(vrai_ecran)
    bouton_croix = Bouton(1540, 10, 50, 50)
    bouton_titre = Bouton(800, 50, 500, 60, coin = False)
    running = True
    flip = False
    clock = pg.time.Clock()
    while running:
        clock.tick(10)
        position = Affichage.convertisseur(pg.mouse.get_pos(), vrai_ecran, ecran)
        clic = None
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                if event.key == pg.K_SPACE or event.key == pg.K_RETURN:
                    flip = not(flip)
            if event.type == pg.MOUSEBUTTONDOWN:
                clic = Affichage.convertisseur(pg.mouse.get_pos(), vrai_ecran, ecran)
        ecran.fill(Noir)
        bouton_croix.dessine(ecran)
        bouton_croix.dessine_icone("Croix.png", ecran)
        bouton_titre.dessine(ecran)
        if not flip:
            Affichage.dessine_matrice(800, 450, image1, ecran, coin = False, dec = True)
            bouton_titre.ecrit_texte(titre1, 50, Noir, ecran)
        else:
            Affichage.dessine_matrice(800, 450, image2, ecran, coin = False, dec = True)
            bouton_titre.ecrit_texte(titre2, 50, Noir, ecran)
        if position in bouton_croix:
            bouton_croix.dessine_contour(Or, 4, ecran)
        if clic:
            if clic in bouton_croix:
                running = False
            else:
                flip = not(flip)
        Affichage.max_blit(vrai_ecran, ecran)


menu()






