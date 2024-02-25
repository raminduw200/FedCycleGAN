# Citation: Wang et al. (2023) FedMed-GAN: Federated domain translation on unsupervised cross-modality brain image
# synthesis. Neurocomputing, 546, 126282. DOI: https://doi.org/10.1016/j.neucom.2023.126282

from collections import OrderedDict

__all__ = ['send_from_server_to_client']


def send_from_server_to_client(server, clients):
    server_state = server.state_dict()
    for k, client in enumerate(clients):
        local_state = client.state_dict()
        for key in server.state_dict().keys():
            local_state[key] = server_state[key]

        client.load_state_dict(local_state)
