# T-AIA-902-TLS_1 Artificial Intelligence Taxi Driver
Reinforcement learning (Python)
Ce dépôt Git contient nos projets d'apprentissage par renforcement Frozen Lake (Bootstrap) et Taxi Driver (projet).

Pour installer les dépendances nescessaire, exécutez :
<code>pip3 install -r requirements.txt</code>.

### Collaborateurs
* Benjamin CASSE
* Gabriel CHASTAING
* Adrien GAVAZZI
* Pauline VIRAULT

### Organisation
* [Trello](https://trello.com/b/1nZnKP87/t-ia-902)
* [GoogleDOC](https://docs.google.com/document/d/1z7EWbaaR_pSQrSnprRgmZyjn0wBgjqF8g2uVHQMobMg/edit?usp=sharing)

### Framework / Bibliothèque
* [Gym](https://gym.openai.com/)
* [MatplotLib](https://matplotlib.org/)
* [Numpy](https://numpy.org/)

## Bootstrap : Frozen Lake

Ce bootstrap vise à entraîner une IA capable de naviguer sur un lac gelé sans tomber dans les tous. Nous avons utilisé l'environnement FrozenLake de la bibliothèque OpenAI Gym pour réaliser ce bootstrap.

#### Description : 

Dans cet environnement, la surface du lac est représentée par une grille où chaque case peut être :
- une case de départ (S),
- une case de glace normale (F),
- une case trou (H),
- ou une case arrivée (G).

Le but de l'agent est de se rendre de la case de départ (S) à la case arrivée (G) en évitant les trous (H). La disposition des cases dans la grille est aléatoire, ce qui signifie que la position des trous et des autres cases peut varier à chaque épisode.

#### Plan :
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"

#### Actions :
Il existe quatre actions possibles dans cet environnement :
- #0#: #Gauche#,
- #1#: #Bas#,
- #2#: #Droite#,
- #3#: #Haut#.

#### Récompenses :
Chaque action entreprise par l'agent se traduira par une récompense :
- Se déplacer sur une case de glace normale (F) entraîne une récompense de 0.
- Atteindre la case d'arrivée (G) entraîne une récompense de +1, constituant la seule récompense positive disponible.
- Tomber dans un trou (H) ou se déplacer en dehors des limites de la grille ne donne pas de récompense supplémentaire, mais l'épisode se termine.

#### Observations :
Dans l'environnement 'FrozenLake-v1' de Gym, l'état est codé de la manière suivante :
- La position actuelle de l'agent dans la grille (4x4 grille) : il y a 16 états possibles. Les états sont numérotés de 0 à (n*n-1), où n est la dimension de la grille

## Taxi Driver

Ce projet vise à former une IA capable de ramasser un passager à un emplacement spécifique et de le transporter vers une autre destination sur une carte. Nous avons utilisé l'environnement Taxi de la bibliothèque OpenAI Gym pour réaliser ce projet. 

#### Description : 
Dans ce contexte, quatre emplacements différents sont définis par les lettres R (red), B (blue), Y (yellow) et G (green). Au début de chaque épisode, le taxi démarre depuis une position aléatoire, tandis que le passager est également placé aléatoirement dans l'un des quatre emplacements (R, B, Y ou G). Le but du taxi est de se rendre à la position du passager, de le prendre en charge, de le transporter vers sa destination (un autre emplacement aléatoire) et de le déposer là-bas. Une fois que le passager est déposé, l'épisode se termine. Il est important de noter que le taxi ne peut pas traverser les murs.

#### Plan :
        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+

#### Actions :
Il existe six actions possibles dans cet environnement : 
- *0*: *Sud* (*bas*),
- *1*: *Nord* (*haut*),
- *2*: *Est* (*droite*),
- *3*: *Ouest* (*gauche*),
- *4*: *Ramassage*,
- *5*: *Dépôt*.

#### Récompenses :
Chaque action entreprise par l'agent se traduira par une récompense :
- Se déplacer entraîne une récompense de -1, ce qui oriente l'agent vers des trajets plus efficaces entre les objectifs.
- Le fait de retirer ou de déposer le passager dans un endroit interdit entraîne une récompense de -10, incitant l'agent à éviter les actions inappropriées de dépôt et de retrait.
- Déposer le passager au bon endroit entraîne une récompense de +20, encourageant ainsi l'agent à atteindre l'objectif, constituant la seule récompense positive disponible.

#### Observations :
Dans l'environnement 'Taxi-v3' de Gym, l'état est codé de la manière suivante : 
- Position du Taxi (5x5 grille) : il y a 25 position possibles pour le taxi.
- Emplacement du Passager : Le passager peut être à 5 positions différentes (quatre emplacements fixes plus dans le taxi).
- Destination : La destination peut être l'un des quatre emplacements fixes.

Les état sont numérotés de 0 à 499, car il y a 25 x 5 x 4 = 500 états possibles. 

Exemple pour Décomposer un état donné comme 134 on décompose en : 
- Position du Taxi : le taxi est à la position (2, 4) dans la grille
- Emplacement du Passager : le passager est à l'emplacement 3 (par exemple, à une certaine position fixe)
- Destination : la destination est l'emplacement 1

