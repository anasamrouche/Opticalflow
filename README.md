# Opticalflow

Module écrit en Rust pour la détection de mouvements avec la fonctionnelle d'énergie de Horn-Schunck par diverses méthodes (Gauss-Seidel simple et pyramidal, descente de gradient).
## Installation

Pour compiler le module, vous pouvez installer la chaîne de compilation de Rust.

### Chaîne de compilation

#### Unix
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### Windows
[Téléchargez le fichier .exe sur le site rustup.](https://rustup.rs/)

### Compilation
Dans votre environnement virtuel lancez :
```bash
cargo add maturin
maturin develop --release
```

En théorie cargo devrait installer les dépendances à partir du fichier cargo.toml.

## Usage
Vous n'avez plus qu'à installer dans votre environnement virtuel les modules du script final et à le lancer. Il produira les vidéos de test organisés comme suit :
```bash
.
└── tests/
    ├── norm_L1/
    │   └── gradient_results
    └── norm_L2/
        ├── gradient_results
        ├── gauss_seidel_results
        └── pyramidal_gauss_seidel_results
```

## Les méthodes supportées pour l'instant
Pour l'instant la descente de gradient a été implémentée pour les normes L1 et L2 et est la seule dans ce cas. Celles basées sur Gauss-Seidel optimisent seulement la fonctionnelle basée sur la norme L2.
