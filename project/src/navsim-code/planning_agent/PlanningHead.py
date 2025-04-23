import torch
import os, glob, re
from datetime import datetime
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

def _auto_resume_path(pattern="checkpoint_epoch*.pth"):
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    epochs = []
    for p in ckpts:
        m = re.search(r"checkpoint_epoch(\d+)\.pth", p)
        if m:
            epochs.append((int(m.group(1)), p))
    return max(epochs, key=lambda x: x[0])[1] if epochs else None

class PlanningHead(nn.Module):
    def __init__(self, ijep_dim, ego_dim, hidden_dim, output_dim):
        super().__init__()
        input_dim = ijep_dim + ego_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.num_poses = output_dim // 3

    def forward(self, visual_features, ego_features):
        x = torch.cat([visual_features, ego_features], dim=1)
        flat = self.mlp(x)
        return flat.view(-1, self.num_poses, 3)

    def fit(
        self,
        dataloader,
        ijepa_encoder,
        device,
        epochs: int,
        lr: float,
        optimizer=None,
        criterion=None,
        save_dir: str = ".",
        resume_from: str = None,
        checkpoint_interval: int = 1,
        use_cls_token: bool = False,
    ):
        os.makedirs(save_dir, exist_ok=True)
        self.to(device)
        optimizer = AdamW(self.parameters(), lr=lr) if optimizer is None else optimizer
        criterion = nn.L1Loss() if criterion is None else criterion

        start_epoch = 1
        if resume_from is None:
            resume_from = _auto_resume_path(os.path.join(save_dir, "checkpoint_epoch*.pth"))
        if resume_from and os.path.exists(resume_from):
            ck = torch.load(resume_from, map_location=device)
            self.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["opt_state"])
            start_epoch = ck["epoch"] + 1
            print(f"Resumed from epoch {ck['epoch']}")

        history = []
        try:
            for epoch in range(start_epoch, epochs + 1):
                self.train()
                total_loss = 0.0
                count = 0
                pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
                for batch in pbar:
                    if batch is None:
                        continue
                    imgs, ego, gt = [x.to(device) for x in batch]
                    with torch.no_grad():
                        out = ijepa_encoder(pixel_values=imgs)
                        feats = (out.pooler_output
                                 if use_cls_token and hasattr(out, "pooler_output")
                                 else out.last_hidden_state.mean(1))
                    pred = self(feats, ego)
                    loss = criterion(pred, gt)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    count += 1
                    pbar.set_postfix(loss=total_loss/count)
                avg = total_loss/count if count else float("nan")
                history.append(avg)
                print(f"Epoch {epoch} → avg loss: {avg:.4f}")

                if epoch % checkpoint_interval == 0:
                    ckpt = {
                        "epoch": epoch,
                        "model_state": self.state_dict(),
                        "opt_state": optimizer.state_dict(),
                    }
                    path = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth")
                    torch.save(ckpt, path)

        except Exception:
            ckpt = {
                "epoch": epoch,
                "model_state": self.state_dict(),
                "opt_state": optimizer.state_dict(),
            }
            path = os.path.join(save_dir, f"checkpoint_failure_epoch{epoch}.pth")
            torch.save(ckpt, path)
            print(f"Interrupted at epoch {epoch}, saved {path}")
            raise

        # final save
        final_loss = history[-1] if history else float("nan")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"planning_head_{ts}_loss{final_loss:.4f}.pth"
        final_path = os.path.join(save_dir, fname)
        torch.save(self.state_dict(), final_path)
        print(f"Saved final weights to {final_path}")

        # clean old checkpoints
        for ck in glob.glob(os.path.join(save_dir, "checkpoint_epoch*.pth")):
            os.remove(ck)
        print("Cleaned up intermediate checkpoints.")

        return history