import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "bev_multimae"
PYTHON_VERSION = "3.11.14"

# Project commands

# Preprocessing commands

# Run and save depth image (high and feature resolution) on a single frame 
@task(help={'folder': "Path to the folder containing images"})
def depth_img(
    ctx: Context, folder="data/raw/camera/kitti_10_imgs", 
    plot_save_folder="reports/figures/depth_imgs"
    ) -> None:
    """Takes single frame from the specified folder and creates feature and depth map."""
    ctx.run(f"uv run src/{PROJECT_NAME}/preprocessing/camera/depth.py {folder} {plot_save_folder}", echo=True, pty=not WINDOWS)

@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", 
        echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

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
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
