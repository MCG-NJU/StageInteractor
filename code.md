# adamixer-code structure

Our code is based on mmdetection and follows the default structure of mmdetection.

AdaMixer related config files can be found in `configs/adamixer/` folder.

AdaMixer related model code includes:
``` shell
mmdet/models/roi_heads/adamixer_decoder.py
### AdaMixer decoder general framework

mmdet/models/roi_heads/bbox_heads/adamixer_decoder_stage.py
### AdaMixer decoder stage depicted in Figure 4

mmdet/models/roi_heads/bbox_heads/adaptive_mixing_operator.py
### implementation of adaptive channel and spatial mixing

mmdet/models/roi_heads/bbox_heads/msaq.py
### 3D sampling implementation

mmdet/models/dense_heads/query_generator.py
### initial query generator
```

You can resort to `Makefile` targets to train models.

Note that some variable names here are temporary names for developing, which might slightly differ from names in the paper. 