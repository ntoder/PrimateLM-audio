# YAMNet (vendored)

`yamnet.py`, `features.py`, `params.py` and `yamnet_class_map.csv` are copied from
[tensorflow/models/research/audioset/yamnet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
under the Apache 2.0 license (see `LICENSE`). Only change: `yamnet.py` uses a relative
import (`from . import features`) so it works as part of our `model` package.

The pretrained weights (`yamnet.h5`, 15 MB) are not committed; `model/embed.py`
downloads them on first use from https://storage.googleapis.com/audioset/yamnet.h5.
