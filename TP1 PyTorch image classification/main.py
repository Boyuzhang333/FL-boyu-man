import torch
from args import parse_args
from get_dataloader import get_dataloader
from get_model import get_model

def metric(loader, model, fn_perte, device, name_experiment, name_model):
    perte = 0
    n_elements = 0
    precision = 0
    with torch.no_grad(): # Non considération sur le calcul de gradient
        for inputs, labels in loader:
            taille = len(labels)
            if name_experiment == "mnist" and (name_model == "linear" or name_model == "mlp"):
                inputs= inputs.view(-1,784).to(device)
            else:
                raise NotImplementedError(
                    "Pas encore implémenter"
                )
            labels = labels.to(device)
            prediction = model(inputs)
            perte += fn_perte(prediction, labels) * taille
            n_elements = n_elements + taille
            _, predicted_indice = torch.max(prediction, axis=1)
            precision += torch.sum(predicted_indice == labels)

    return perte/n_elements, precision/n_elements

if __name__ == '__main__':
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    args = parse_args()
    print(args.experiment, args.batch_size, args.model, args.device)
    try:
        assert str(args.device) == str(device)
    except AssertionError as e:
        print(f'Error: torch device {device}, not the same as argument device {args.device}')
        raise e
    ## Remplir les codes dans get_model
    train_loader, test_loader = get_dataloader(args.experiment, args.batch_size)
    model = get_model(args.model, args.experiment, args.device)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ## Question 4: Écrire la boucle d'entraînement
    ## Question 6: Métriques d'entraînement -> metric(..) à chaque époque 

    ### Your work below ### 
    for e in range(args.epochs):
        raise NotImplementedError
