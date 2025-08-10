import numpy as np
from utils import AverageMeterSet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.distributions import Categorical
from tqdm import tqdm
import torch.nn.functional as F
import os
from torch.nn.utils import clip_grad_norm_

def negative_sample(sampler, sample_num):
    neg_ids = sampler.sample((sample_num,))
    return neg_ids + 1 + 3

def gather_indexes(sequence_tensor, positions):
    """
    Gathers the vectors at the specific positions over a minibatch.
    """
    batch_size, seq_length, width = sequence_tensor.size()
    flat_offsets = torch.arange(0, batch_size, dtype=torch.long) * seq_length
    flat_offsets = flat_offsets.unsqueeze(-1)  # reshape to [batch_size, 1]
    flat_positions = (positions + flat_offsets).view(-1)
    flat_sequence_tensor = sequence_tensor.view(batch_size * seq_length, width)
    output_tensor = flat_sequence_tensor.index_select(0, flat_positions)
    return output_tensor

class PyTorchAdamWeightDecayOptimizer(AdamW):
    """A basic Adam optimizer that includes L2 weight decay for PyTorch."""
    def __init__(self, params, learning_rate, weight_decay_rate=0.01,
                 beta1=0.9, beta2=0.999, epsilon=1e-6):
        """Constructs a AdamWeightDecayOptimizer for PyTorch."""
        super().__init__(params, lr=learning_rate, betas=(beta1, beta2),
                         eps=epsilon, weight_decay=weight_decay_rate)


class BERT4ETHTrainer(nn.Module):
    def __init__(self, args, vocab, model, data_loader):
        super(BERT4ETHTrainer, self).__init__()
        self.args = args
        self.device = args.device
        self.vocab = vocab
        self.model = model.to(self.device)

        self.data_loader = data_loader
        self.num_epochs = args.num_epochs

        # Parameters for pre-training task, not related to the model
        self.dense = nn.Linear(args.hidden_size, args.hidden_size).to(self.device)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12).to(self.device)
        # self.output_bias = torch.nn.Parameter(torch.zeros(1+args.neg_sample_num, device=self.device))
        self.optimizer, self.lr_scheduler = self._create_optimizer()

    def calculate_loss(self, batch):
        input_ids = batch["input_ids"]
        counts = batch["counts"]
        values = batch["values"]
        io_flags = batch["io_flags"]
        positions = batch["positions"]
        input_mask = batch["input_mask"]
        labels = batch["labels"]  # [B, T] (masked는 정수, 나머지는 -1)

        # 1. Forward BERT
        h = self.model(input_ids, counts, values, io_flags, positions)  # [B, T, H]

        # 2. Transform for prediction
        input_tensor = self.dense(h)
        input_tensor = self.transform_act_fn(input_tensor)
        input_tensor = self.LayerNorm(input_tensor)  # [B, T, H]

        # 3. Output logits using token embedding weights (tied weights)
        output_weights = self.model.embedding.token_embed.weight  # [V, H]
        logits = torch.matmul(input_tensor, output_weights.T)  # [B, T, V]

        # 4. Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),     # [B*T, V]
            labels.view(-1),                      # [B*T]
            ignore_index=-1                       # -1은 loss에서 무시
        )
        return loss


    def train(self):
        assert self.args.ckpt_dir, "must specify the directory for storing checkpoint"
        accum_step = 0
        for epoch in range(self.num_epochs):
            # print("bias:", self.output_bias[:10])
            accum_step = self.train_one_epoch(epoch, accum_step)
            if (epoch+1) % 5 == 0 or epoch==0:
                self.save_model(epoch+1, self.args.ckpt_dir)

    def load(self, ckpt_dir):
        self.model.load_state_dict(torch.load(ckpt_dir))

    def infer_embedding(self, ):
        self.model.eval()
        tqdm_dataloader = tqdm(self.data_loader)
        embedding_list = []
        address_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm_dataloader):
                 # 1. address는 str로 직접 사용
                addresses = batch["address_str"]  # List[str]

                # 2. 나머지 입력은 device로 보내기
                input_ids = batch["input_ids"].to(self.device)
                counts = batch["counts"].to(self.device)
                values = batch["values"].to(self.device)
                io_flags = batch["io_flags"].to(self.device)
                positions = batch["positions"].to(self.device)

                # 3. 모델 forward
                h = self.model(input_ids, counts, values, io_flags, positions)  # [B, T, H]
                cls_embedding = h[:, 0, :]  # [B, H]

                # mean embedding
                # mean_embedding = torch.mean(h, dim=1)
                embedding_list.append(cls_embedding)
                address_list += addresses  # 그대로 리스트로 누적
        
        # 5. 전체 시퀀스 기준 임베딩 반환
        embedding_tensor = torch.cat(embedding_list, dim=0).cpu()  # [N, H]
        address_array = np.array(address_list)  # [N], str array

        return address_array, embedding_tensor.numpy()


    def train_one_epoch(self, epoch, accum_step):
        self.model.train()

        tqdm_dataloader = tqdm(self.data_loader)
        
        # 1. 에폭의 총 손실을 누적할 변수 초기화
        total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()}

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 5.0)  # Clip gradients

            self.optimizer.step()
            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()

            accum_step += 1
            
            # 2. 배치별 손실을 total_loss에 누적
            total_loss += loss.item()
            
            tqdm_dataloader.set_description(
                'Epoch {}, Step {}, loss {:.6f} '.format(epoch+1, accum_step, loss.item()))
        
        # 3. 에폭이 끝난 후 평균 손실 계산
        avg_loss = total_loss / len(self.data_loader)
        
        # 4. 평균 손실 출력
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")

        return accum_step


    def save_model(self, epoch, ckpt_dir):
        print(ckpt_dir)
        os.makedirs(f'mlm/{ckpt_dir}', exist_ok=True)
        ckpt_dir = os.path.join(f'mlm/{ckpt_dir}', "epoch_" + str(epoch)) + ".pth"
        print("Saving model to:", ckpt_dir)
        torch.save(self.model.state_dict(), ckpt_dir)

    def _create_optimizer(self):
        """Creates an optimizer training operation for PyTorch."""
        num_train_steps = self.args.num_train_steps
        num_warmup_steps = self.args.num_warmup_steps
        for name, param in self.named_parameters():
            print(name, param.size(), param.dtype)

        optimizer = PyTorchAdamWeightDecayOptimizer([
            {"params": self.parameters()}
        ],
            learning_rate=self.args.lr,
            weight_decay_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-6
        )

        # Implement linear warmup and decay
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda step: min((step + 1) / num_warmup_steps, 1.0)
                                                      if step < num_warmup_steps else (num_train_steps - step) / (
                                                                  num_train_steps - num_warmup_steps))

        return optimizer, lr_scheduler

