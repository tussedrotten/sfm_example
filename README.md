# sfm-example
An sfm-example using [gtsam](https://github.com/borglab/gtsam) for BA, based on feature tracks from the Holmenkollen dataset.

## Dependencies
You can install all dependencies in [requirements.txt](requirements.txt) with pip:
```bash
pip install -r requirements.txt
```

## Examples
- [explore_matches_data.py](explore_matches_data.py): Visualizes the matches in the dataset.
- [explore_map_data.py](explore_map_data.py): Shows the dataset in 3D.
- [incremental_sfm.py](incremental_sfm.py): Interactive SfM example solved using factor graphs with [gtsam](https://github.com/borglab/gtsam).
