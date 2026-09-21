# ELON.MD — The 5-Step Engineering & Architecture Algorithm

Ce document formalise l'algorithme d'ingénierie et d'optimisation en 5 étapes (inspiré des principes d'Elon Musk). Il sert de **grille d'audit et de décision absolue** pour chaque revue de code, chaque proposition d'architecture et chaque évolution de l'application.

L'ordre des 5 étapes est **strict, immuable et non négociable**.

---

## 1. Remettre en question chaque prérequis (*Question every requirement*)

- **Aucune contrainte n'est sacrée** : Ne jamais accepter une exigence technique, produit ou ergonomique comme une vérité absolue, même (et surtout) si elle émane d'un "expert", d'une bibliothèque tierce ou d'une convention établie.
- **Trouver la source réelle** : Chaque contrainte doit être rattachée à une justification technique vérifiable et mesurable, jamais à une justification vague (*"c'est comme ça qu'on fait d'habitude"*).
- **Les 500 "Why"** : Pousser le questionnement jusqu'aux limites de la physique et du matériel (mémoire Metal unifiée, bande passante mémoire, coût des allocations, réalité du GPU).
- **Bannir le code spéculatif (YAGNI extrême)** : Si une fonctionnalité, une abstraction ou un paramètre est là "au cas où", son prérequis est considéré invalide et doit être rejeté.

---

## 2. Supprimer tout ce qui peut l'être (*Delete any part of the process you can*)

- **Supprimer avant de modifier** : Si une étape, un module, une abstraction, un paramètre d'API ou un élément d'interface ne peut être justifié de façon irréfutable à l'étape 1, **le supprimer immédiatement**.
- **La règle des 10 %** : Si vous n'êtes jamais contraint de faire marche arrière pour réintroduire au moins 10 % de ce que vous avez supprimé, c'est que vous n'avez pas supprimé assez de code.
- **Moins de pièces = Moins de pannes** : Tout composant ou ligne de code qui n'existe pas ne peut pas planter, ne consomme pas de mémoire et n'engendre aucune dette de maintenance.

---

## 3. Simplifier et optimiser (*Simplify and optimize*)

- **Ne jamais optimiser ce qui ne devrait pas exister** : L'erreur n°1 de l'ingénieur est de passer du temps à accélérer, tuner ou raffiner une fonction, un composant ou une couche d'abstraction qui aurait simplement dû être effacée à l'étape 2.
- **Repartir d'une page blanche** : Reconstruire le chemin le plus court et le plus direct entre l'entrée utilisateur et l'exécution matérielle (GPU/Metal/disque) avec uniquement les éléments indispensables ayant survécu.
- **Rejeter l'over-engineering** : Privilégier le code plat, direct et explicite plutôt que les architectures abstraites prématurées ou les wrappers inutiles.

---

## 4. Accélérer les temps de cycle (*Accelerate cycle time*)

- **Viser les ordres de grandeur** : Ne pas chercher 5 % d'amélioration marginale, chercher des facteurs $2\times$, $5\times$, $10\times$ ou $100\times$ (ex : quantification UNet 4-bit vs fp16, cache d'embeddings de texte, I/O mémoire sans copie).
- **Feedback loop ultra-court** : La boucle entre l'action utilisateur et le résultat doit être quasi instantanée.
- **Échouer vite pour pivoter vite** : Réduire le temps d'inférence, de build, de test et de démarrage de l'application au minimum vital pour tester des hypothèses à vitesse maximale.

---

## 5. Automatiser (*Automate*)

- **Automatiser en TOUT DERNIER recours uniquement** : Ne jamais ajouter de daemons de fond, de workers complexes, de files de traitement asynchrones ou d'automatismes lourds avant d'avoir rigoureusement franchi les étapes 1 à 4.
- **Le piège fatal** : Automatiser trop tôt fige dans le marbre et accélère un processus inutile, inefficace ou mal conçu.
- **Automatisation ciblée** : Quand les 4 premières étapes sont validées, automatiser uniquement ce qui élimine les frictions répétitives réelles.

---

## 📋 Grille d'Audit Rapide pour toute Revue

Lors de chaque audit ou demande de review, appliquer systématiquement cette séquence :

1. **Questionner** : Quels paramètres, boutons, modèles ou endpoints existent sans réelle nécessité matérielle ou utilisateur ?
2. **Supprimer** : Quels fichiers, wrappers, options inutilisées ou branches de code pouvons-nous jeter dès maintenant ?
3. **Simplifier** : Comment réduire le code restant à sa forme la plus pure, directe et minimaliste ?
4. **Accélérer** : Quel est le goulot d'étranglement matériel majeur et comment gagner un facteur $\times 5$ ou $\times 10$ ?
5. **Automatiser** : Quelle tâche manuelle résiduelle mérite réellement d'être automatisée maintenant que le flux est minimal ?
