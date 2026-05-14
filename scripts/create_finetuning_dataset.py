from bev_multimae.preprocessing.camera.depth import DepthEstimator

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main():
    de = DepthEstimator(cfg, device='cuda', plot=cfg.plotting)
    de._load_model()

    