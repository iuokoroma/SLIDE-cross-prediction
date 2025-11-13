# SLIDE-cross-prediction
Cross prediction for SLIDE (Significant Latent Factor Interaction Discovery and Exploration)

### What to run in your environment (requires an R installation to run rpy2)
```
yaml_path = 'path/to/yaml_params.yaml'
val_X_path = 'path/to/val_X.csv'
val_Y_path = 'path/to/val_Y.csv'

SLIDE_cp = show_cross_prediction(yaml_path, val_X_path, val_Y_path)
SLIDE_cp
```
