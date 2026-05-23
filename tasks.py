import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "bev_multimae"
PYTHON_VERSION = "3.11.14"


def py(tmp=False):
    return f"/tmp/venv_{os.environ['USER']}/bin/python" if tmp else "uv run"


def run(ctx: Context, cmd: str, tmp=False) -> None:
    ctx.run(f"{py(tmp)} {cmd}", echo=True, pty=not WINDOWS)


# Project commands

# PREPROCESSING COMMANDS

# Run and save depth image (high and feature resolution) on a single frame 
@task(help={'folder': "Path to the folder containing images"})
def depth_img(
    ctx: Context, folder="data/raw/camera/front_right", 
    plot_save_folder="reports/figures/depth_imgs", tmp=False
    ) -> None:
    """Takes single frame from the specified folder and creates feature and depth map."""
    run(ctx, f"src/{PROJECT_NAME}/preprocessing/camera/depth.py {folder} {plot_save_folder}", tmp)

# Either extract mcap 
@task
def read_mcap(ctx: Context, l=False, t=False, tmp=False) -> None:
    """Read MCAP file. Use -l to list topics, -t to list transforms, default extracts."""
    flag = "list_topics" if l else "list_transforms" if t else "extract"
    run(ctx, f"src/{PROJECT_NAME}/preprocessing/mcap_reader.py {flag}", tmp)
    
@task
def preprocess_data(ctx: Context, tmp=False) -> None:
    """Preprocess data."""
    run(ctx, f"src/{PROJECT_NAME}/data.py data/raw data/processed", tmp)

# TRAINING COMMANDS
@task
def train_model(ctx: Context, mode="finetune", tmp=False) -> None:
    """Run training. mode=pretrain or finetune."""
    if mode == "pretrain":
        config = "config"
    elif mode == "finetune":
        config = "config_finetune"
    else:
        raise ValueError("mode must be 'pretrain' or 'finetune'")

    run(ctx, f"scripts/train.py --config-name {config} train_mode={mode}", tmp)

@task
def finetune_model_pass(ctx: Context, tmp=False) -> None:
    """Run finetune model pass."""
    run(ctx, "scripts/finetune_model_pass.py --config-name config_finetune", tmp)

@task
def pretrain_model_pass(ctx: Context, tmp=False) -> None:
    """Run pretraining model pass."""
    run(ctx, "scripts/single_pred.py --config-name config", tmp)

@task
def test(ctx: Context, tmp=False) -> None:
    """Run tests."""
    run(ctx, "coverage run -m pytest tests/", tmp)
    run(ctx, "coverage report -m -i", tmp)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task
def build_docs(ctx: Context, tmp=False) -> None:
    """Build documentation."""
    run(ctx, "mkdocs build --config-file docs/mkdocs.yaml --site-dir build", tmp)

@task
def serve_docs(ctx: Context, tmp=False) -> None:
    """Serve documentation."""
    run(ctx, "mkdocs serve --config-file docs/mkdocs.yaml", tmp)