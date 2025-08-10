import numpy as np
import torch
import math
import random
import torch.utils.data as data_utils
import copy


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
        seed = args.dataloader_random_seed
        # self.rng = random.Random(seed)
        self.rng = random.Random()
        self.max_predictions_per_seq = math.ceil(self.args.max_seq_length * self.args.masked_lm_prob)

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):

        tranxs = copy.deepcopy(self.seq_list[index])
        address = tranxs[0][0]  # 대표 주소 (시퀀스 소유자)

        # address → identity 매핑
        address2id = {}
        identity_seq = []
        id_counter = 3
        for tx in tranxs:
            addr = tx[0]
            if addr not in address2id:
                address2id[addr] = id_counter
                id_counter += 1
            identity_seq.append(address2id[addr])

        cand_indexes = list(range(1, len(identity_seq)))
        self.rng.shuffle(cand_indexes)
        num_to_predict = min(self.max_predictions_per_seq,
                         max(1, int(len(tranxs) * self.args.masked_lm_prob)))
        
        labels = [-1 for _ in range(len(identity_seq))]  # -1 = not masked
        num_masked = 0
        covered_indexes = set()

        for idx in cand_indexes:
            if num_masked >= num_to_predict:
                break
            if idx in covered_indexes:
                continue
            covered_indexes.add(idx)

            labels[idx] = identity_seq[idx]  # 정답 저장
            identity_seq[idx] = self.vocab.convert_tokens_to_ids(["[MASK]"])[0]  # MASK 토큰 ID
            num_masked += 1


        # MAP discrete feature to int
        block_timestamps = list(map(lambda x: x[2], tranxs))
        values = list(map(lambda x: x[3], tranxs))
        io_flags = list(map(map_io_flag, tranxs))
        counts = list(map(lambda x: x[5], tranxs))
        positions = convert_timestamp_to_position(block_timestamps)
        input_mask = [1] * len(identity_seq)

        assert max(identity_seq) < 103, f"[input_ids] OOB: {max(identity_seq)} >= 103"
        assert max(counts) < 15, f"[counts] OOB: {max(counts)}"
        assert max(values) < 15, f"[values] OOB: {max(values)}"
        assert max(io_flags) < 3, f"[io_flags] OOB: {max(io_flags)}"
        assert max(positions) < 100, f"[positions] OOB: {max(positions)}"

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

         # 검증
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
        # mask_prob, mask_token, max_predictions_per_seq):
        self.args = args
        self.seq_list = seq_list
        self.vocab = vocab
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):

        # only one index as input
        tranxs = self.seq_list[index]
        address = tranxs[0][0]

        # address → identity 매핑
        address2id = {}
        identity_seq = []
        id_counter = 4
        for tx in tranxs:
            addr = tx[0]
            if addr not in address2id:
                address2id[addr] = id_counter
                id_counter += 1
            identity_seq.append(address2id[addr])

        # MAP discrete feature to int
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