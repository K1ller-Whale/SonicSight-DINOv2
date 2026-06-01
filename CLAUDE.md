# Project Conventions

## Python Environment
Always use this Python executable for running scripts and tests:
C:/Users/H/AppData/Local/Programs/Python/Python312/python.exe

## PyTorch Patterns
- Always move tensors to device explicitly: `tensor = tensor.to(self.device)`
- Use `torch.no_grad()` context manager when running frozen modules (DINOv2)
- Prefer `einops.rearrange` over manual `.view()` or `.reshape()` for clarity
- Always assert tensor shapes in unit tests: `assert out.shape == torch.Size([...])`
- Use `torch.testing.assert_close` for numerical comparisons in tests
- Complex spectrograms are always 2-channel real/imaginary: shape [B, 2, F, T]

## Lightning Structure
Always structure LightningModule like this:
```python
class SeparatorModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = Separator(cfg.model)
        self.loss_fn = PITLoss(cfg.loss)

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn(self.model(batch))
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss_fn(self.model(batch))
        self.log("val/sisnri", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), 
                                  lr=self.hparams.cfg.train.lr,
                                  weight_decay=1e-4)
```

## Hydra Config Schema
Always structure configs like this:
```yaml
# configs/model.yaml
model:
  n_sources: 4
  audio_channels: [2, 32, 64, 128, 256, 512]
  attention:
    n_heads: 8
    n_layers: 2
    d_model: 512
    ffn_dim: 2048
    dropout: 0.1
  visual:
    backbone: facebook/dinov2-base
    patch_dim: 768
    proj_dim: 512
```

## What NOT to do
- NEVER use torchaudio.functional.spectrogram directly — always use torchaudio.transforms.Spectrogram
- NEVER batch DINOv2 calls across sources — process each source independently
- NEVER recompute cRM targets on the fly — always load from cache
- NEVER use BatchNorm in the U-Net — always use GroupNorm with groups=8
- NEVER skip writing a unit test before moving to the next component
- NEVER use .view() for tensor reshaping — use einops.rearrange instead
- NEVER use bias=True in the visual projection linear layer

## Shell and Path Conventions
- ALWAYS use forward slashes in all paths: C:/Users/H/... not C:\Users\H\...
- NEVER use backslashes in any bash command or file path
- When running Python use this exact path with forward slashes:
  C:/Users/H/AppData/Local/Programs/Python/Python312/python.exe
- NEVER use the `type` command to read file output — use `cat` instead
- When reading command output files always use: cat /path/to/file

## API Usage
- Do not make multiple small sequential tool calls — batch related 
  operations into single larger steps where possible
- Write complete file contents in one call, not line by line
- Do not re-read files you have already read in this session
- Keep responses concise — no lengthy explanations, just code and results

## On API Failures
- If you get a provider error, wait 30 seconds and retry the exact same step
- Do not restart from scratch, just retry