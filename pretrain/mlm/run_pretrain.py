import pickle as pkl

from config import args
from modeling import BERT4ETH
from vocab import SimpleVocab

from mlm.dataloader import BERT4ETHDataloader
from mlm.mlm_trainer import BERT4ETHTrainer


def train():
    print("===========Load Sequence===========")
    with open(args.data_dir + "eoa2seq_" + args.bizdate + ".pkl", "rb") as f:
        eoa2seq = pkl.load(f)

    print("number of target user account:", len(eoa2seq))

    # prepare dataset
    vocab = SimpleVocab()

    # dataloader
    dataloader = BERT4ETHDataloader(args, vocab, eoa2seq)
    train_loader = dataloader.get_train_loader()

    # model
    model = BERT4ETH(args)

    # tranier
    trainer = BERT4ETHTrainer(args, vocab, model, train_loader)
    trainer.train()


if __name__ == '__main__':
    train()

