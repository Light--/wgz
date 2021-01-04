import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader


def embeddings_Gen():
    with torch.no_grad():
        m = RunManager()
        for run in RunBuilder.get_runs(params):
            if run.model == "tripletNet":
                if run.network == "sqe":
                    embedding_net = SQEEmbNet()
                    embedding_net.load_state_dict(
                        torch.load(
                            "./models/10_tripletNet_sqe_10_adam_0001Final_Scheduler.pth.tar"
                        )["state_dict"]
                    )
                elif run.network == "alex":
                    embedding_net = AlexEmbNet()
                    embedding_net.load_state_dict(
                        torch.load(
                            "./models/10_tripletNet_alex_10_adam_0001Final_Scheduler.pth.tar"
                        )["state_dict"]
                    )
                elif run.network == "resnet":
                    embedding_net = ResnetEmbNet()
                    embedding_net.load_state_dict(
                        torch.load(
                            "./models/10_tripletNet_resnet_10_adam_0001Final_Scheduler.pth.tar"
                        )["state_dict"]
                    )
                elif run.network == "effB0":
                    embedding_net = EffNetEmbNet()
                    embedding_net.load_state_dict(
                        torch.load(
                            "./models/10_tripletNet_effB0_10_adam_001OnlyEffB1.pth.tar"
                        )["state_dict"]
                    )
                elif run.network == "effB3":
                    embedding_net = EffNetEmbNet()
                    embedding_net.load_state_dict(
                        torch.load(
                            "./models/10_tripletNet_effB0_10_adam_001OnlyEffB1.pth.tar"
                        )["state_dict"]
                    )
            elif run.model == "siameseNet":
                if run.network == "sqe":
                    embedding_net = SQEEmbNet()
                    embedding_net.load_state_dict(
                        torch.load(
                            "./models/10_siameseNet_sqe_10_adam_0001Final_Scheduler.pth.tar"
                        )["state_dict"]
                    )
                elif run.network == "alex":
                    embedding_net = AlexEmbNet()
                    embedding_net.load_state_dict(
                        torch.load(
                            "./models/10_siameseNet_alex_10_adam_0001Final_Scheduler.pth.tar"
                        )["state_dict"]
                    )
                elif run.network == "resnet":
                    embedding_net = ResnetEmbNet()
                    embedding_net.load_state_dict(
                        torch.load(
                            "./models/10_siameseNet_resnet_10_adam_0001Final_Scheduler.pth.tar"
                        )["state_dict"]
                    )
                elif run.network == "effB0":
                    embedding_net = EffNetEmbNet()
                    embedding_net.load_state_dict(
                        torch.load(
                            "./models/10_siameseNet_effB0_10_adam_0001Final_Scheduler.pth.tar"
                        )["state_dict"]
                    )
                elif run.network == "effB3":
                    embedding_net = EffNetEmbNet()
                    # embedding_net.load_state_dict(torch.load('./models/10_tripletNet_effB0_10_adam_0001Final_Scheduler.pth.tar')['state_dict'])
                    embedding_net.load_state_dict(
                        torch.load(
                            "./models/10_tripletNet_effB0_10_adam_001OnlyEffB1.pth.tar"
                        )["state_dict"]
                    )
            embedding_net = embedding_net.to(device)
            Dset = datasetGen()
            data_loader = DataLoader(Dset, batch_size=len(Dset), shuffle=True)
            m.begin_run(run, embedding_net)
            plt.figure(figsize=(10, 10))
            batch = next(iter(data_loader))
            img_embs = []
            labels = []
            for i in range(len(batch[1])):
                embedding_net.eval()
                img = batch[0][i].to(device)
                label = batch[1][i]
                img_emb = embedding_net(img.unsqueeze(0))
                img_embs.append(img_emb)
                labels.append(label)
            img_embs = torch.stack(img_embs).reshape(-1, 32)
            m.tb.add_embedding(img_embs, labels, batch[0])
            m.end_run()
