import pickle as pkl
import torch

from config import args
from modeling import BERT4ETH
from vocab import SimpleVocab

from cluster.xent_dataloader import XentTrainDataLoader
from cluster.cluster_trainer import BERT4ETHXentTrainer


def train():
    print("===========Load Sequence===========")
    with open(args.data_dir + "eoa2seq_" + args.bizdate + ".pkl", "rb") as f:
        eoa2seq = pkl.load(f)

    print("number of target user account:", len(eoa2seq))

    # dataloader
    dataloader = XentTrainDataLoader(args, eoa2seq)
    train_loader = dataloader.get_xent_loader()

    # model
    model = BERT4ETH(args)
    # ✅ MLM 학습된 가중치 로드
    mlm_ckpt_path = "./mlm/cpkt_local/epoch_50.pth"
    print(f"Loading MLM pretrained model from {mlm_ckpt_path}")
    state_dict = torch.load(mlm_ckpt_path)
    model.load_state_dict(state_dict, strict=False)

    # tranier
    trainer = BERT4ETHXentTrainer(args, model, train_loader)
    trainer.train()


if __name__ == '__main__':
    train()

