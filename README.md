### Experiment Types/Approaches
- Final Conv and FC classification Layers (learnable) with Siamese and Triplet Architectures.
- Concating LPIPS_like features extracted from [low, mid and higher] layers with Final Classifier {_*Next Step*_}

### Modules and Classes init
- network
    - AlexEmbNet
    - ResnetEmbNet
    - SQEEmbNet
    <!-- - ResNextEmbNet -->
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
        epoch 
        train
        test
- infer {_*Next Step*_}
        

### Hyperparameters
- Learning Rate = [.01, .001, .0001]
- Batch Size = [10, 25, 100]
- Model = ['siameseNet', 'tripletNet']
- Network = ['sqe','alex', 'resnet']
- ~lpips_like = [True, False] {_*Next Step*_}
- Contrastive and Triplet Losses Margin = [0.5, 1., 1.5]

