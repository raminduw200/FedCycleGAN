import flwr as fl
from models import create_model
from options.train_options import TrainOptions

# opt = TrainOptions().parse()  # get training options
# server = create_model(opt, id=0, is_client=False)  # create a model given opt.model and other options
# server.setup(opt)


strategy = fl.server.strategy.FedAvg(
#     initial_parameters=server.get_parameters(),
)

fl.server.start_server(
    server_address="localhost:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=2)
)


# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=fc / ac,
#     fraction_eval=0.2,  # not used - no federated evaluation
#     min_fit_clients=fc,
#     min_eval_clients=2,  # not used
#     min_available_clients=ac,
#     eval_fn=get_eval_fn(generator, discriminator, g_ema, g_optim, d_optim, args),
#     on_fit_config_fn=fit_config,
#     on_evaluate_config_fn=evaluate_config,
#     initial_parameters=fl.common.weights_to_parameters(model_weights),
# )
#
# fl.server.start_server("0.0.0.0:8081", config={"num_rounds": rounds}, strategy=strategy)
