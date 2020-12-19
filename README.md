### Experiment Types/Approaches
- Final Conv and FC classification Layers (learnable) with Siamese and Triplet Architectures. {*Done*}
- Concating LPIPS_like features extracted from [low, mid and higher] layers with Final Classifier {_*Next Step*_}

### Modules and Classes init
- network
    - AlexEmbNet
    - ResnetEmbNet
    - SQEEmbNet.
    - EfficientNet/det
    - ResNextEmbNet
    - SiameseNet
    - TripletNet
- custom datasets
    - SiamDset
    - TripDset
- losses
    - Constrastive Loss
    - Triplet Loss
- train
    -main
- infer {_*Next Step*_}
- Embeddings Generator and Projection {_*Next Step*_}

### Hyperparameters
- Learning Rate = [.1, .01]
- Batch Size = [5, 10, 15, 20, 25, 30]
- Model = ['siameseNet', 'tripletNet']
- Network = ['sqe','alex', 'resnet']
- ~lpips_like = [True, False] {_*Next Step*_}
- ~~Contrastive and Triplet Losses Margin = [0.5, 1., 1.5]~~

