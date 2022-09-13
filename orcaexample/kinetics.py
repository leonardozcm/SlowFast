import argparse
from slowfast.models import build_model
import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
from slowfast.datasets import loader
from slowfast.utils.parser import load_config

from bigdl.orca.learn.pytorch import Estimator
from torch.utils.data import DataLoader
from bigdl.orca import init_orca_context, stop_orca_context


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The cluster mode, such as local, yarn-client, yarn-cluster, spark-submit or k8s.')
parser.add_argument('--runtime', type=str, default="spark",
                    help='The runtime backend, one of spark or ray.')
parser.add_argument('--cfg_files', type=str, default="configs/Kinetics/SLOWFAST_8x8_R50.yaml",
                    help='The path to config file')
parser.add_argument(
    "--opts",
    help="See slowfast/config/defaults.py for all options",
    default=None,
    nargs=argparse.REMAINDER,
)
parser.add_argument('--backend', type=str, default="bigdl",
                    help='The backend of PyTorch Estimator; bigdl, ray, and spark are supported')
args = parser.parse_args()
if args.runtime == "ray":
    init_orca_context(runtime=args.runtime, address=args.address)
else:
    if args.cluster_mode == "local":
        init_orca_context(memory="12g", cores=8)
    elif args.cluster_mode.startswith("yarn"):
        if args.cluster_mode == "yarn-client":
            init_orca_context(cluster_mode="yarn-client")
        elif args.cluster_mode == "yarn-cluster":
            init_orca_context(cluster_mode="yarn-cluster", memory=args.executor_memory, driver_memory=args.driver_memory)
    elif args.cluster_mode == "spark-submit":
        init_orca_context(cluster_mode="spark-submit")

class WrappedDataLoader(DataLoader):
    def __init__(self, loader, func) -> None:
        data_loader_args = {
            "dataset": loader.dataset,
            "batch_size": loader.batch_size,
            "shuffle": False,
            "num_workers": loader.num_workers,
            "collate_fn": func(loader.collate_fn),
            "pin_memory": loader.pin_memory,
            "drop_last": loader.drop_last,
            "timeout": loader.timeout,
            "worker_init_fn": loader.worker_init_fn
        }
        super().__init__(**data_loader_args)

def reduceWrapper(func):
    def reduceIterElement(batch):
        batch=func(batch)
        assert len(batch)==5, "Should yield inputs, labels, index, time, meta in dataloader"
        return batch[0], batch[1]
    return reduceIterElement

def train_loader_creator(config, batch_size):
    train_loader = loader.construct_loader(config, "train")
    loader.shuffle_dataset(train_loader, 0)
    train_loader.collate_fn = reduceWrapper(train_loader.collate_fn)
    # return WrappedDataLoader(train_loader, reduceWrapper)
    return train_loader

def model_creator(config):
    return build_model(config)

def optim_creator(model, config):
    return optim.construct_optimizer(model, config)

def loss_creator(config):
    return losses.get_loss_func(config.MODEL.LOSS_FUNC)(reduction="mean")

cfg = load_config(args, args.cfg_files)

loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

if args.backend == "bigdl":
    net = model_creator(cfg)
    optimizer = optim_creator(model=net, config=cfg)
    orca_estimator = Estimator.from_torch(model=net,
                                          optimizer=optimizer,
                                          loss = loss_fun,
                                          backend=args.backend,
                                          config=cfg)
    orca_estimator.fit(data=train_loader_creator(cfg, 0),
                       epochs=cfg.SOLVER.MAX_EPOCH
                       )
elif args.backend in ["ray", "spark"]:
    print("cfg keys",dict(cfg).keys)
    orca_estimator = Estimator.from_torch(model=model_creator,
                                          optimizer=optim_creator,
                                          loss=loss_creator,
                                          backend=args.backend,
                                          config=cfg,
                                          use_tqdm=True)
    orca_estimator.fit(data=train_loader_creator, epochs=cfg.SOLVER.MAX_EPOCH)

stop_orca_context()
