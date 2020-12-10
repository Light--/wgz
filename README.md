### Experiment Types/Approaches
- Final Conv and FC classification Layers (learnable) with Siamese and Triplet Architectures.
- LPIPS like *Next Step*
- Concating LPIPS_like features extracted from [low, mid and higher] layers with Final Classifier *Next Step*

### modules Structure
- network
    - AlexEmbNet
    - ResnetEmbNet
    - SQEEmbNet
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
        

### Hyperparameters
- Learning Rate = [.01, .001, .0001]
- Batch Size = [10, 25, 100]
- Architecture Type = ['siameseNet', 'tripletNet']
- Embedding Nets = ['sqe','alex', 'resnet']
- ~lpips_like = [True, False] *Next Step*
- Contrastive and Triplet Losses Margin = [0.5, 1., 1.5]

