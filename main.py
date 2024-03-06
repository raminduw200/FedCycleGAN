# import pickle
# from pathlib import Path

# import hydra
# from hydra.core.hydra_config import HydraConfig
# from omegaconf import DictConfig, OmegaConf

import flwr as fl

from data import create_dataset
from options.train_options import TrainOptions
from client_cyclegan import generate_client_fn


# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
# @hydra.main(config_path="conf", config_name="base", version_base=None)
# def main(cfg: DictConfig):
def main():
    # print(OmegaConf.to_yaml(cfg))
    # save_path = HydraConfig.get().runtime.output_dir

    opt = TrainOptions().parse()  # get training options
    custom_dataloader = create_dataset(opt)

    client_fn = generate_client_fn(opt, custom_dataloader.dataloaders)

    strategy = fl.server.strategy.FedAvg(
        # fraction_fit=0.0,  # in simulation, since all clients are available at all times, we can just use
        # # `min_fit_clients` to control exactly how many clients we want to involve during fit
        # min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
        # fraction_evaluate=0.0,  # similar to fraction_fit, we don't need to use this argument.
        # min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        # min_available_clients=cfg.num_clients,  # total clients in the simulation
        # on_fit_config_fn=get_on_fit_config(
        #     cfg.config_fit
        # ),  # a function to execute to obtain the configuration to send to the clients during fit()
        # evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )  # a function to run on the server side to evaluate the global model.

    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=opt.client_count,
        # cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            # num_rounds=cfg.num_rounds
            num_rounds=opt.fl_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        strategy=strategy,  # our strategy of choice
        client_resources={
            # "num_cpus": 2,
            "num_gpus": 0.5,
        },
    )

    # results_path = Path(save_path) / "results.pkl"

    # results = {"history": history, "anythingelse": "here"}

    # with open(str(results_path), "wb") as h:
    #     pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
