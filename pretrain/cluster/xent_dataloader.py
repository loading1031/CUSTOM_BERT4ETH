import random
import torch
import torch.utils.data as data_utils
import copy
from mlm.dataloader import map_io_flag, convert_timestamp_to_position


class XentTrainDataLoader(data_utils.Dataset):
    def __init__(self, args, address2seqs):
        self.address2seqs = {k: v for k, v in address2seqs.items() if len(v) >= 2}
        self.address_list = list(self.address2seqs.keys())
        self.args = args
        self.rng = random.Random(args.dataloader_random_seed)

    def __len__(self):
        return len(self.address_list)

    def __getitem__(self, index):
        addr = self.address_list[index]
        full_seq = self.address2seqs[addr]  # 주소의 전체 트랜잭션 시퀀스 (1차원 리스트)

        # NT-Xent를 위해 전체 시퀀스에서 두 개의 다른 서브 시퀀스(뷰)를 만듭니다.
        # 이 두 서브 시퀀스가 positive pair가 됩니다.
        pos_pair = self.get_two_subsequences(full_seq)

        batch = []
        for sub_seq in pos_pair:
            # 각 서브 시퀀스를 전처리 함수에 전달합니다.
            batch.append(self.preprocess_subsequence(addr, sub_seq))

        return batch  # [positive1, positive2]
    
    def get_two_subsequences(self, full_seq):
        max_num_tokens = self.args.max_seq_length - 1
        seq_len = len(full_seq)

        # 서브 시퀀스를 두 개 생성합니다.
        sub_seq1 = self.get_sliding_window_subsequence(full_seq, seq_len, max_num_tokens)
        sub_seq2 = self.get_sliding_window_subsequence(full_seq, seq_len, max_num_tokens)

        return [sub_seq1, sub_seq2]

    def get_sliding_window_subsequence(self, full_seq, seq_len, max_num_tokens):
        # BERT4ETH와 동일한 슬라이딩 윈도우 로직을 적용하여 서브 시퀀스를 추출합니다.
        if seq_len < max_num_tokens / 0.4:
            start_idx = 0
        elif seq_len < max_num_tokens:
            sliding_step = round(self.args.max_seq_length * 0.4)
            start_idx = self.rng.choice(range(0, max(seq_len - max_num_tokens + 1, 1), sliding_step))
        else:
            sliding_step = round(self.args.max_seq_length * 0.6)
            start_idx = self.rng.choice(range(0, max(seq_len - max_num_tokens + 1, 1), sliding_step))

        return full_seq[start_idx:start_idx + max_num_tokens]
    
    def preprocess_subsequence(self, address, sub_seq):
        sub_seq = copy.deepcopy(sub_seq)
        CLS_TOKEN_ID = 2

        # 헤더를 추가하여 최종 시퀀스를 만듭니다.
        final_seq = [[CLS_TOKEN_ID, 0, 0, 0, 0, 0]] + sub_seq

        # 기존 로직과 동일하게 identity 매핑을 수행합니다.
        address2id = {}
        identity_seq = []
        id_counter = 3
        for tx in final_seq:
            addr_item = tx[0]
            if addr_item not in address2id:
                address2id[addr_item] = id_counter
                id_counter += 1
            identity_seq.append(address2id[addr_item])

        labels = [-1] * len(identity_seq)

        block_timestamps = list(map(lambda x: x[2], final_seq))
        values = list(map(lambda x: x[3], final_seq))
        io_flags = list(map(map_io_flag, final_seq))
        counts = list(map(lambda x: x[5], final_seq))
        positions = convert_timestamp_to_position(block_timestamps)
        input_mask = [1] * len(identity_seq)

        max_seq_length = self.args.max_seq_length
        pad = lambda x, v: x + [v] * (max_seq_length - len(x))

        identity_seq = pad(identity_seq, 0)
        values = pad(values, 0)
        io_flags = pad(io_flags, 0)
        counts = pad(counts, 0)
        positions = pad(positions, 0)
        input_mask = pad(input_mask, 0)
        labels = pad(labels, -1)

        return {
            # "address_str": address,
            "input_ids": torch.LongTensor(identity_seq),
            "counts": torch.LongTensor(counts),
            "values": torch.LongTensor(values),
            "io_flags": torch.LongTensor(io_flags),
            "positions": torch.LongTensor(positions),
            "input_mask": torch.LongTensor(input_mask),
            "labels": torch.LongTensor(labels),
            "address_ids": torch.LongTensor([self.address_list.index(address)])
        }

    def get_xent_loader(self):
        dataset = self
        dataloader = data_utils.DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=12, # CPU 코어 수 (많으면 발열 심함 주의)
            collate_fn=xent_collate_fn
        )
        return dataloader


def xent_collate_fn(batch):
    flat = [sample for pair in batch for sample in pair]
    batched = {k: torch.stack([d[k] for d in flat], dim=0) for k in flat[0].keys()}
    return batched
