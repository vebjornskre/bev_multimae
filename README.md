# BEV-MultiMAE

A unified masked autoencoder architecture for fusing BEV radar representations and camera features projected into BEV space, inspired by Voxel-MAE, BEV-MAE, and MultiMAE.

## Dependancies
If you use ZoeDepth as the depth model then you have to activate the zoe env:
source .venv_zoe/bin/activate

## Project structure (structure currently changes a lot now so this is not properly updated yet)

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── raw/
│   │   ├── radar/
│   │   └── camera/
│   └── processed/
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── tools/                    # Entry scripts
│   ├── train_pretrain.py
│   ├── train_finetune.py
│   ├── evaluate.py
│   └── infer.py
├── src/                      # Source code
│   ├── bev_multimae/
│   │   ├── __init__.py
|   |   ├── datasets/
│   │   |   ├── radar.py
│   │   │   ├── camera.py
│   │   │   └── paired.py
│   │   ├── preprocessing/    # Sensor to BEV transforms
│   │   │   ├── radar/
│   │   │   │   ├── voxelize.py
│   │   │   │   └── to_bev.py
│   │   │   ├── camera/
│   │   │   │   ├── depth.py
│   │   │   │   ├── lift.py
│   │   │   │   └── softsplat.py
│   │   │   └── geometry.py
│   │   ├── multimae/         # Architecture code
│   │   │   ├── model.py
│   │   │   ├── encoder/
│   │   │   ├── decoders/
│   │   │   ├── adapters/
│   │   │   ├── masking/
│   │   │   └── losses/
│   │   ├── engines/          # Training logic
│   │   │   ├── pretrain.py
│   │   │   ├── finetune.py
│   │   │   ├── inference.py
│   │   │   └── evaluate.py
│   │   ├── utils/
│   │   └── visualization/
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


## Acknowledgements

This project structure is based on the MLOps template by Nicki Skafte Detlefsen:
https://github.com/SkafteNicki/mlops
````
