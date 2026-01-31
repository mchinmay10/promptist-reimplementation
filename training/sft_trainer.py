import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class SFTTrainer:
    def __init__(
        self,
        model,
        dataset,
        lr: float = 5e-5,
        batch_size: int = 4,
        num_epochs: int = 1,
        output_dir: str = "checkpoints/sft",
        device: str | None = None,
    ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        os.makedirs(output_dir, exist_ok=True)

    def train(self):
        self.model.train()

        global_step = 0

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            epoch_loss = 0.0

            for batch in tqdm(self.dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

            avg_loss = epoch / len(self.dataloader)
            print(f"Average loss: {avg_loss:.4f}")

            self.save_checkpoint(epoch)

            # Temporary Debug
            print("\nSample generation:")
            print(self.model.generate("a cat sitting on a chair"))
            valid = (labels != -100).sum().item()
            print("Supervised tokens:", valid)

    def save_checkpoint(self, epoch: int):
        ckpt_path = os.path.join(
            self.output_dir,
            f"epoch_{epoch + 1}",
        )

        self.model.model.save_pretrained(ckpt_path)
        self.model.tokenizer.save_pretrained(ckpt_path)

        print(f"Checkpoint saved to {ckpt_path}")
