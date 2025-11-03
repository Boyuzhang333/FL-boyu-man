import torch


def get_model(name_model, name_experiment, device):
    if name_model == "linear":
        # Question 3: modèle linéaire https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        ### Your work below ### 
        model = None 
        raise NotImplementedError
        ### Your work above ###
        for param in model.parameters():
            print(param.shape)
    elif name_model == "mlp":
        class MLP(torch.nn.Module):
            def __init__(self):
                super(MLP, self).__init__()
                ### Your work below ###
                self.linear1 = torch.nn.Linear(28*28, 1) # remplacer 1 par 64 -> meilleur espace latent 
                self.activation = None # quelle activation utiliser ici ? 
                self.linear2 = torch.nn.Linear(1, 10) # remplacer 1 par 64 -> meilleur espace latent
                ### Your work above ### 
                
            def forward(self, x):
                ## Question 7
                # don't forget to take a look at your init ! 
                ### Your work below ###
                x = None 
                raise NotImplementedError
                ### Your work above ### 
                return x
        model = MLP()

    elif name_model == "cnn":
        pass

    return model.to(device)