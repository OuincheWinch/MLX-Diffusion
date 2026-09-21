================================================================================
MLX-DIFFUSION — Guide de Démarrage, Commandes & Spécifications des Modèles
================================================================================

Studio de génération d'images local optimisé pour Apple Silicon (MLX & Metal GPU).
Backend FastAPI (Python) + Frontend React 19 / Vite.


================================================================================
1. COMMANDES DE DÉMARRAGE ET D'ARRÊT
================================================================================

Les deux services (Backend sur le port 8001 et Frontend sur le port 5174) doivent
être exécutés depuis la racine du projet :
  cd /Volumes/Externe/IA/MLX-DIFFUSION

--------------------------------------------------------------------------------
A) DÉMARRAGE EN AVANT-PLAN (Recommandé en développement)
--------------------------------------------------------------------------------
Ouvrir 2 onglets de terminal :

Onglet 1 (Backend FastAPI) :
  ./dev-backend.sh
  # (ou: cd backend && caffeinate -s ../venv/bin/uvicorn main:app --reload --port 8001)

Onglet 2 (Frontend Vite) :
  cd frontend && npm run dev
  # (ou: cd frontend && caffeinate -s vite)

Pour arrêter : appuyer sur Ctrl + C dans chaque onglet.

--------------------------------------------------------------------------------
B) DÉMARRAGE EN ARRIÈRE-PLAN (Mode Daemon / Headless)
--------------------------------------------------------------------------------
Depuis la racine du projet :

  # 1. Lancer le backend en arrière-plan
  nohup ./dev-backend.sh > /tmp/mlx-backend.log 2>&1 &

  # 2. Lancer le frontend en arrière-plan
  cd frontend && nohup npm run dev > /tmp/mlx-frontend.log 2>&1 &
  cd ..

--------------------------------------------------------------------------------
C) ARRÊT PROPRE DE TOUS LES SERVICES (One-Liner)
--------------------------------------------------------------------------------
Pour couper immédiatement et proprement tous les démons (Backend, Frontend, SDXL Engine) :

  kill $(lsof -ti :8001 -ti :5174) 2>/dev/null
  pkill -f "sdxl_engine" 2>/dev/null

Vérification qu'aucun processus ne subsiste :
  lsof -i :8001 -i :5174
  # (Si cette commande ne renvoie rien, tous les services sont éteints proprement)

--------------------------------------------------------------------------------
D) URLS D'ACCÈS
--------------------------------------------------------------------------------
- Interface Web (Studio) :  http://localhost:5174
- API Backend (Swagger)   :  http://localhost:8001/docs
- Port 8001 obligatoire   :  Le port 8000 est réservé aux autres outils locaux.


================================================================================
2. MODÈLES SUPPORTÉS & AVANTAGES COMPARATIFS
================================================================================

Le studio intègre 4 moteurs d'inférence complémentaires, tous quantifiés et
optimisés pour tirer le maximum d'Apple Silicon (16 GB unifiés) sans OOM Metal :

--------------------------------------------------------------------------------
1. FLUX.2-klein 4B (Moteur Principal — Black Forest Labs DiT)
--------------------------------------------------------------------------------
- Répertoire / ID : flux2-klein-4b (mlx-community/flux2-klein-4b-4bit)
- Étapes idéales  : 4 steps | Guidance : 1.0 (guidance-distilled, pas de CFG négatif)
- Vitesse (M1)    : ~90s en 768×768 | ~120–140s en 1024×1024

AVANTAGES CLÉS :
  ✦ Multi-Références In-Context (1 à 10 images) :
    Permet de conditionner l'image sur 1 à 10 images de référence simultanées
    (conservant cohérence des personnages, style, accessoires) en les référençant
    simplement dans le prompt sous la forme `Image 1`, `Image 2`, etc.
  ✦ Correspondance Exacte des Couleurs (#HEX) :
    Respect strict des codes hexadécimaux (#FF3366, #00E5FF, etc.). Une palette
    visuelle intégrée permet l'insertion directe à l'emplacement du curseur.
  ✦ Décodage PiD (mflux) :
    Option sous les paramètres avancés pour remplacer le VAE standard par le
    super-résolution NVIDIA PiD (suppression d'artefacts & micro-textures).
  ✦ Multi-LoRAs FLUX.2 :
    Prise en charge des fichiers .safetensors compatibles FLUX.2. Changement de
    slider en 0s (sans recharger le modèle).

--------------------------------------------------------------------------------
2. Juggernaut XL Lightning (Moteur SDXL Distillé & Ultra-Rapide)
--------------------------------------------------------------------------------
- Répertoire / ID : juggernaut-xl-lightning (RunDiffusion/Juggernaut-XL-Lightning)
- Exécution       : venv-sdxl (Python 3.14 + MLX natif 4-bit)
- Étapes idéales  : 4 steps | Scheduler : euler_trailing | Guidance : 1.0
- Vitesse (M1)    : ⚡ ~15–20s en 1024×1024 (avec TAESD) | ~50s (VAE Natif)

AVANTAGES CLÉS :
  ✦ ⚡ Génération 4-step distilled SOTA :
    Qualité photographique exceptionnelle sans compromis grâce à l'architecture
    distillée 4 étapes de RunDiffusion.
  ✦ ⚡⚡ Décodeur VAE TAESD ultra-rapide (~0.5s) :
    Remplacement instantané du décodeur lourd par TAESD compilé en MLX pur,
    éliminant 20 secondes d'attente à la fin de la génération.
  ✦ Multi-LoRAs SDXL & Empilement Dynamique :
    Chargement et fusion à la volée des LoRAs Kohya / CivitAI (format SDXL)
    sans altérer les poids quantifiés du modèle de base.
  ✦ Samplers Avancés & Support Negative Prompt :
    euler_trailing (défaut optimisé 4-step), dpmpp_2m_karras, euler_a_substep,
    euler_a, euler, ddim.

--------------------------------------------------------------------------------
3. Krea 2 Turbo 13B (Moteur Photoréaliste Haute Fidélité)
--------------------------------------------------------------------------------
- Répertoire / ID : krea2-turbo (local:krea2-turbo-q4)
- Règle des Steps :
  * SANS LoRA 4-step : DOIT TOURNER EN 8 STEPS (Défaut).
    Sans le LoRA de distillation, 4 steps ne suffisent pas pour converger :
    l'image reste floue, incomplète et bruitée. Le modèle natif requiert
    impérativement 8 steps pour atteindre sa pleine netteté et convergence.
  * AVEC LoRA 4-step : TOURNE EN 4 STEPS.
    Si le LoRA de distillation (`Krea2-Turbo-Distill-4step`) est activé,
    le modèle converge parfaitement en seulement 4 steps, réduisant le temps
    de génération à ~140s au lieu de ~250s.
- Résolution max  : 512×512 (recommandé) ou 512×768 (Portrait — limite mémoire 16GB)
- Vitesse (M1)    : ~140s en 512×512 (4 steps avec LoRA) | ~250s (8 steps natif sans LoRA)

AVANTAGES CLÉS :
  ✦ Esthétique et Photoréalisme de pointe (Modèle 13B SOTA) :
    Rendu bluffant des peaux, matières, lumières volumétriques et détails fins.
  ✦ Quantification Q4 du Text Encoder Qwen3-VL :
    Réduction dynamique de 7.5 GB à ~1.9 GB permettant de faire tourner un
    modèle 13B complet sur Mac 16 GB unifiés.
  ✦ Moteur Anti-SIGABRT :
    Sécurisation des conversions de mémoire C/C++ pour empêcher tout crash
    brutal du processus lors du décodage VAE causal 3D.
  ✦ Règle de convergence 8 steps (natif) vs 4 steps (avec LoRA distillé) :
    Garantit une netteté maximale et évite tout rendu inachevé.

--------------------------------------------------------------------------------
4. Z-Image Turbo 6B (Moteur Rapide Grands Formats)
--------------------------------------------------------------------------------
- Répertoire / ID : z-image-turbo (filipstrand/Z-Image-Turbo-mflux-4bit)
- Étapes idéales  : 9 steps | Guidance : Non applicable
- Vitesse (M1)    : ~40s en 1024×1024

AVANTAGES CLÉS :
  ✦ Idéal pour les esquisses rapides et les panoramas 16:9 (1280×720).
  ✦ Empreinte mémoire ultra-légère.


================================================================================
3. LE BOUTON "✨ ENHANCE" (AMÉLIORATION INTELLIGENTE DU PROMPT)
================================================================================

Situé juste à droite du champ de prompt dans l'interface, le bouton "✨ Enhance"
est un optimiseur de prompt neuronal 100% local, alimenté par un modèle de langage
embarqué (Qwen2.5-0.5B-Instruct-4bit via mlx-lm).

Il fonctionne entièrement hors-ligne sur Apple Silicon (Metal GPU), prend ~0.5 à 1s,
consomme moins de 350 Mo de RAM, et n'envoie aucune donnée vers le cloud.

--------------------------------------------------------------------------------
A) RÈGLES STRUCTURALES GUIDÉES PAR LE MODÈLE (Model-Guided Prompting)
--------------------------------------------------------------------------------
Chaque famille de modèle possède son propre encodeur de texte (T5, dual-CLIP,
Qwen3-VL, DiT). Le bouton Enhance adapte dynamiquement ses règles de réécriture
selon le modèle sélectionné :

1. FLUX.2-klein 4B (Encodeur T5) :
   - Génère une prose narrative naturelle, fluide et descriptive (2 à 3 phrases).
   - Décrit précisément la physique de la lumière (lumière rasante, diffusion douce,
     reflets spéculaires), les textures matérielles réelles et l'optique caméra
     (profondeur de champ, 50mm f/1.4).
   - Proscrit formellement les mots-clés parasites / "tag soup" ("masterpiece",
     "8k", "trending on artstation", "photorealistic").

2. Juggernaut XL Lightning (Encodeurs CLIP-L + OpenCLIP-G) :
   - Structure le prompt avec un vocabulaire cinématographique et photographique
     percutant (medium shot, 35mm photography, volumetric lighting, rim light,
     dramatic shadow contrast).
   - Met en avant la séparation nette sujet/arrière-plan et la composition du cadre.

3. Krea 2 Turbo 13B (Encodeur Multimodal Qwen3-VL) :
   - Privilégie des angles de vue marqués et dynamiques (contre-plongée, grand-angle,
     lignes de fuite), des contours nets et des palettes chromatiques vibrantes.

4. Z-Image Turbo 6B (Transformer DiT) :
   - Accentue le photoréalisme tactile, les micro-textures de surface (pores de
     peau, tissage de tissu, gouttelettes) et la netteté chirurgicale du sujet.

--------------------------------------------------------------------------------
B) PRÉSERVATION STRICTE DES TRIGGERS DE LoRA (LoRA Trigger Preservation)
--------------------------------------------------------------------------------
Lorsque vous activez un ou plusieurs LoRAs dans la section "LoRAs" :

1. Détection automatique :
   Le backend consulte instantanément la base locale des LoRAs (`data/loras.json`)
   et extrait tous les mots-clés d'activation officiels (trigger words) associés
   aux LoRAs actuellement cochés / actifs.

2. Intégration verbatim & naturelle :
   - Si vous avez déjà tapé le trigger dans votre texte, Enhance le préserve
     scrupuleusement à sa place et étoffe harmonieusement ce qui l'entoure.
   - Si vous ne l'avez pas encore tapé, Enhance l'intègre directement et naturellement
     dans la définition du sujet principal.
   - L'ambiance générale générée par Enhance est harmonisée avec le thème du LoRA
     (ex: cyberpunk, rétro, aquarelle, personnage spécifique).

3. Garantie déterministe anti-omission :
   Une étape de vérification post-génération contrôle automatiquement la présence
   exacte de chaque trigger word. Si le modèle de langage a oublié un mot-clé, le
   système le réinsère automatiquement au début du prompt enrichi.

--------------------------------------------------------------------------------
C) DÉCHARGEMENT AUTOMATIQUE DES LoRAs AU CHANGEMENT DE MODÈLE
--------------------------------------------------------------------------------
- Pour éviter les incohérences et les crashs de moteurs (ex: charger un LoRA Krea 2
  ou SDXL dans FLUX.2 ou Z-Image Turbo), l'interface décharge automatiquement tout
  LoRA incompatible dès que vous changez de modèle dans le sélecteur.
- Si le modèle sélectionné ne supporte aucun LoRA ou ne dispose d'aucun LoRA
  compatible dans la bibliothèque, la liste des LoRAs actifs se vide instantanément.
- Le menu déroulant "+ Add LoRA..." ne propose que les LoRAs compatibles avec
  l'architecture en cours d'utilisation.

--------------------------------------------------------------------------------
D) COMMENT L'UTILISER ?
--------------------------------------------------------------------------------
1. Écrivez simplement une idée courte dans le champ Prompt (ex: "a cybernetic cat").
2. (Optionnel) Sélectionnez un ou plusieurs LoRAs compatibles dans la section LoRAs.
3. Cliquez sur le bouton "✨ Enhance".
4. En ~1 seconde, votre prompt est remplacé par une description experte, enrichie,
   adaptée au modèle d'inférence en cours et contenant tous vos déclencheurs LoRA !


================================================================================
4. OPTIONS AVANCÉES & FONCTIONNALITÉS STUDIO
================================================================================

- Studio Split-Screen  : Formulaire à gauche, grand Canvas interactif à droite.
- Bouton "✨ Enhance"   : Amélioration neuronale locale 1-clic avec adaptation modèle
                         et conservation garantie des triggers LoRA.
- Upscaleur Neuronal   : Bouton "✨ AI Neural 2x" (SeedVR2 latent 1-step) et boutons
                         "⚡ Fast 2x" / "⚡ Fast 4x" (Lanczos haute fidélité en 0.18s).
- Variantes 1-Clic     : Boutons "Dice" (randomize), "+1 Seed" et "+1024 Batch Seed".
- Galerie Lazy-Loaded  : Tri par date, tags personnalisés, recherche textuelle et
                         miniatures WebP/PNG générées à la demande.
- Drag & Drop Direct   : Glisser-déposer une image ou un fichier .safetensors dans
                         le formulaire pour l'ajouter automatiquement aux références
                         ou enregistrer le LoRA.
- Réutilisation Réf    : Le bouton "🖼️ Use as Reference" dans le Canvas ou la Galerie
                         envoie directement l'image dans le plateau multi-références.
- Gestion Concurrence  : File d'attente FIFO unifiée avec option de bascule ou
                         d'annulation instantanée en cas de nouvelle requête.
================================================================================
