import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm

def nt_xent_loss(embeddings, address_ids, temperature=0.1):
    """
    Computes the NT-Xent loss. This revised version correctly handles the denominator.

    Args:
        embeddings (torch.Tensor): Tensor of shape [B, D].
        address_ids (torch.Tensor): Tensor of shape [B], with integer IDs.
        temperature (float): The temperature parameter.

    Returns:
        torch.Tensor: The mean NT-Xent loss for the batch.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # Normalize embeddings for cosine similarity
    # Note: Assuming embeddings are already normalized from the cluster_head,
    # but re-normalizing for safety.
    proj = F.normalize(embeddings, dim=-1)

    # Cosine similarity matrix [B, B]
    sim_matrix = torch.matmul(proj, proj.T)
    
    # Scale similarities by temperature
    sim_matrix = sim_matrix / temperature

    # Create a mask for positive pairs
    # address_ids_expanded shape becomes [1, B]
    address_ids_expanded = address_ids.unsqueeze(0)
    
    # pos_mask shape becomes [B, B]
    # It's True where address_ids match
    pos_mask = torch.eq(address_ids_expanded, address_ids.unsqueeze(1)).float().to(device)

    # A mask to remove self-similarity (the diagonal)
    self_mask = torch.eye(batch_size, device=device).float()
    
    # Remove self-similarity from the positive mask
    # This leaves only true positive pairs where i != j
    pos_mask = pos_mask - self_mask
    
    # Exponentials of similarities
    exp_sims = torch.exp(sim_matrix)

    # Numerator: Sum of exponentials of positive similarities
    # This multiplies exp_sims with the mask for positive pairs (excluding self)
    numerator = (exp_sims * pos_mask).sum(dim=1)
    
    # Denominator: Sum of exponentials of ALL similarities, excluding self-similarity
    # We sum all exp_sims and subtract the diagonal (self-similarity)
    denominator = exp_sims.sum(dim=1) - exp_sims.diag()

    # To prevent division by zero or log(0)
    numerator = numerator.clamp(min=1e-8)
    denominator = denominator.clamp(min=1e-8)
    
    # Calculate NT-Xent loss
    loss = -torch.log(numerator / denominator)
    return loss.mean()


class BERT4ETHXentTrainer(nn.Module):
    def __init__(self, args, model, data_loader):
        super().__init__()
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.optimizer, self.lr_scheduler = self._create_optimizer()
        self.num_epochs = args.num_epochs

    def train(self):
        assert self.args.ckpt_dir, "Checkpoint directory must be set"
        accum_step = 0

        for epoch in range(self.num_epochs):
            accum_step = self.train_one_epoch(epoch, accum_step)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.save_model(epoch + 1, self.args.ckpt_dir)
    
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
                '[XENT]Epoch {}, Step {}, loss {:.6f} '.format(epoch+1, accum_step, loss.item()))
        
        # 3. 에폭이 끝난 후 평균 손실 계산
        avg_loss = total_loss / len(self.data_loader)
        
        # 4. 평균 손실 출력
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")

        return accum_step

    def calculate_loss(self, batch):
        input_ids = batch["input_ids"]
        counts = batch["counts"]
        values = batch["values"]
        io_flags = batch["io_flags"]
        positions = batch["positions"]
        
        # Squeeze the address_ids tensor to flatten its shape from [B, 1] to [B]
        address_ids = batch["address_ids"].squeeze(1)

        h = self.model(input_ids, counts, values, io_flags, positions)  # [B, T, H]
        cls_embedding = h[:, 0, :]  # [B, H]
        proj = self.model.cluster_head(cls_embedding)  # [B, D]
        # The nt_xent_loss function will handle normalization again, but it's okay.
        
        return nt_xent_loss(proj, address_ids, temperature=0.1)

    def save_model(self, epoch, ckpt_dir):
        os.makedirs(f'cluster/{ckpt_dir}', exist_ok=True)
        ckpt_path = os.path.join(f'cluster/{ckpt_dir}', f"epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), ckpt_path)

    def _create_optimizer(self):
        optimizer = AdamW(self.model.parameters(),
                          lr=self.args.lr,
                          weight_decay=0.01,
                          betas=(0.9, 0.999),
                          eps=1e-6)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min((step + 1) / self.args.num_warmup_steps, 1.0)
            if step < self.args.num_warmup_steps else
            (self.args.num_train_steps - step) / max(1, (self.args.num_train_steps - self.args.num_warmup_steps))
        )

        return optimizer, scheduler

