import numpy as np
import torch
import math
import random
import torch.utils.data as data_utils
import copy
from typing import Dict, List, Any


def map_io_flag(tranxs):
    flag = tranxs[4]
    if flag == "OUT":
        return 1
    elif flag == "IN":
        return 2
    else:
        return 0

def convert_timestamp_to_position(block_timestamps):
    position = [0]
    if len(block_timestamps) <= 1:
        return position
    last_ts = block_timestamps[1]
    idx = 1
    for b_ts in block_timestamps[1:]:
        if b_ts != last_ts:
            last_ts = b_ts
            idx += 1
        position.append(idx)
    return position


# BERT4ETHDataloader 클래스
class BERT4ETHDataloader:

    def __init__(self, args, vocab, eoa2seq):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.eoa2seq = eoa2seq
        self.vocab = vocab
        self.seq_list = self.preprocess(eoa2seq)

    def preprocess(self, eoa2seq):
        self.masked_lm_prob = self.args.masked_lm_prob
        # self.rng = random.Random(self.args.dataloader_random_seed)
        self.rng = random.Random()
        self.sliding_step = round(self.args.max_seq_length * 0.6)

        # preprocess
        length_list = []
        for eoa in eoa2seq.keys():
            seq = eoa2seq[eoa]
            length_list.append(len(seq))

        length_list = np.array(length_list)
        print("Median:", np.median(length_list))
        print("Mean:", np.mean(length_list))
        print("Seq num:", len(length_list))

        # clip
        max_num_tokens = self.args.max_seq_length - 1
        seqs = []
        idx = 0
        for eoa, seq in eoa2seq.items():
            if len(seq) <= max_num_tokens:
                seqs.append([[eoa, 0, 0, 0, 0, 0]])
                seqs[idx] += seq
                idx += 1
            elif len(seq) > max_num_tokens:
                beg_idx = list(range(len(seq) - max_num_tokens, 0, -1 * self.sliding_step))
                beg_idx.append(0)

                if len(beg_idx) > 500:
                    beg_idx = list(np.random.permutation(beg_idx)[:500])
                    for i in beg_idx:
                        seqs.append([[eoa, 0, 0, 0, 0, 0]])
                        seqs[idx] += seq[i:i + max_num_tokens]
                        idx += 1

                else:
                    for i in beg_idx[::-1]:
                        seqs.append([[eoa, 0, 0, 0, 0, 0]])
                        seqs[idx] += seq[i:i + max_num_tokens]
                        idx += 1

        self.rng.shuffle(seqs)
        return seqs

    def get_train_loader(self):
        dataset = BERT4ETHTrainDataset(self.args, self.vocab, self.seq_list)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           num_workers=12, #TODO: 학습 전 CPU 코어 확인
                                           shuffle=True, pin_memory=True)
        return dataloader

    def get_eval_loader(self):
        dataset = BERT4ETHEvalDataset(self.args, self.vocab, self.seq_list)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.eval_batch_size,
                                           num_workers=12, # 학습 전 CPU 코어 확인
                                           shuffle=False, pin_memory=True)

        return dataloader


class BERT4ETHTrainDataset(data_utils.Dataset):

    def __init__(self, args, vocab, seq_list):
        # mask_prob, mask_token, max_predictions_per_seq):
        self.args = args
        self.seq_list = seq_list
        self.vocab = vocab
        self.rng = random.Random()
        self.max_predictions_per_seq = math.ceil(self.args.max_seq_length * self.args.masked_lm_prob)
        self.MASK_TOKEN_ID = self.vocab.get_mask_id() # MASK 토큰 ID는 1
        self.PAD_TOKEN_ID = self.vocab.get_pad_id() # PAD 토큰 ID는 0
        self.CLS_TOKEN_ID = self.vocab.get_cls_id() # CLS 토큰 ID는 2

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        tranxs = copy.deepcopy(self.seq_list[index])
        address = tranxs[0][0]  # 대표 주소 (시퀀스 소유자)

         # --- 핵심 변경사항: 시퀀스 내에서만 유효한 address2id 매핑 생성 ---
        address_counts = {}
        for tx in tranxs[1:]:
            addr = tx[0]
            address_counts[addr] = address_counts.get(addr, 0) + 1
            
        sorted_addresses = sorted(address_counts, key=address_counts.get, reverse=True)
        
        address2id = {}
        # [PAD]=0, [MASK]=1, [CLS]=2. 시퀀스 내부 주소는 3부터 시작
        id_counter = 3
        for addr in sorted_addresses:
            address2id[addr] = id_counter
            id_counter += 1

        # 첫 번째 항목은 [CLS] 토큰이므로, 주소 매핑에서 제외
        identity_seq = [self.CLS_TOKEN_ID]
        for tx in tranxs[1:]:
            addr = tx[0]
            try:
                identity_seq.append(address2id[addr])
            except KeyError:
                raise KeyError(f"Address '{addr}' from sequence was not found in the global address mapping.")

        cand_indexes = list(range(1, len(identity_seq)))
        self.rng.shuffle(cand_indexes)
        num_to_predict = min(self.max_predictions_per_seq,
                             max(1, int(len(tranxs[1:]) * self.args.masked_lm_prob)))
        
        labels = [-1 for _ in range(len(identity_seq))]
        num_masked = 0
        covered_indexes = set()

        for idx in cand_indexes:
            if num_masked >= num_to_predict:
                break
            if idx in covered_indexes:
                continue
            covered_indexes.add(idx)

            labels[idx] = identity_seq[idx]
            identity_seq[idx] = self.MASK_TOKEN_ID # MASK 토큰 ID 사용
            num_masked += 1

        # MAP discrete feature to int (tranxs[1:] 사용)
        block_timestamps = list(map(lambda x: x[2], tranxs))
        values = list(map(lambda x: x[3], tranxs))
        io_flags = list(map(map_io_flag, tranxs))
        counts = list(map(lambda x: x[5], tranxs))
        positions = convert_timestamp_to_position(block_timestamps)
        input_mask = [1] * len(identity_seq)

        # padding
        max_seq_length = self.args.max_seq_length
        pad = lambda x, v: x + [v] * (max_seq_length - len(x))

        identity_seq = pad(identity_seq, 0)
        values = pad(values, 0)
        io_flags = pad(io_flags, 0)
        counts = pad(counts, 0)
        positions = pad(positions, 0)
        input_mask = pad(input_mask, 0)
        labels = pad(labels, -1)

        assert all(len(x) == max_seq_length for x in [identity_seq, values, io_flags, counts, positions, input_mask, labels])

        return {
            "address_str": address, 
            "input_ids": torch.LongTensor(identity_seq),
            "counts": torch.LongTensor(counts),
            "values": torch.LongTensor(values),
            "io_flags": torch.LongTensor(io_flags),
            "positions": torch.LongTensor(positions),
            "input_mask": torch.LongTensor(input_mask),
            "labels": torch.LongTensor(labels),
        }


class BERT4ETHEvalDataset(data_utils.Dataset):

    def __init__(self, args, vocab, seq_list):
        self.args = args
        self.seq_list = seq_list
        self.vocab = vocab
        self.rng = random.Random(args.dataloader_random_seed)
        self.PAD_TOKEN_ID = self.vocab.get_pad_id()
        self.CLS_TOKEN_ID = self.vocab.get_cls_id()

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        tranxs = self.seq_list[index]
        address = tranxs[0][0]

        # --- 핵심 변경사항: 시퀀스 내에서만 유효한 address2id 매핑 생성 ---
        address_counts = {}
        for tx in tranxs[1:]:
            addr = tx[0]
            address_counts[addr] = address_counts.get(addr, 0) + 1
            
        sorted_addresses = sorted(address_counts, key=address_counts.get, reverse=True)
        
        address2id = {}
        # [PAD]=0, [MASK]=1, [CLS]=2. 시퀀스 내부 주소는 3부터 시작
        id_counter = 3
        for addr in sorted_addresses:
            address2id[addr] = id_counter
            id_counter += 1

        identity_seq = [self.CLS_TOKEN_ID]
        for tx in tranxs[1:]:
            addr = tx[0]
            try:
                identity_seq.append(address2id[addr])
            except KeyError:
                raise KeyError(f"Address '{addr}' from sequence was not found in the global address mapping.")

        block_timestamps = list(map(lambda x: x[2], tranxs))
        values = list(map(lambda x: x[3], tranxs))
        io_flags = list(map(map_io_flag, tranxs))
        counts = list(map(lambda x: x[5], tranxs))
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

        assert all(len(x) == max_seq_length for x in [identity_seq, values, io_flags, counts, positions, input_mask])

        return {
            "address_str": address, 
            "input_ids": torch.LongTensor(identity_seq),
            "counts": torch.LongTensor(counts),
            "values": torch.LongTensor(values),
            "io_flags": torch.LongTensor(io_flags),
            "positions": torch.LongTensor(positions),
            "input_mask": torch.LongTensor(input_mask),
        }