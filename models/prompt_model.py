import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from models.tokenizer import format_prompt


class PromptOptimizer(torch.nn.Module):
    def __init__(self, model_name: str, tokenizer):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = tokenizer

        # Resize embeddings for added special tokens
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        user_prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.9,
        top_p: float = 0.95,
    ) -> str:
        device = next(self.parameters()).device

        formatted = format_prompt(user_prompt)

        inputs = self.tokenizer(formatted, return_tensors="pt").to(device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return decoded.split("Optimized:")[-1].strip()

    @torch.no_grad()
    def sample(
        self,
        user_prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.9,
        top_p: float = 0.95,
    ):
        device = next(self.parameters()).device

        formatted = format_prompt(user_prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Generate token IDs (only the new ones)
        gen_ids = outputs.sequences[:, input_len:]

        # Compute log-probability
        logprobs = []
        for step, logits in enumerate(outputs.scores):
            logp = F.log_softmax(logits, dim=-1)
            token_id = gen_ids[:, step]
            logprobs.append(logp.gather(-1, token_id.unsqueeze(-1)))

        logprob = torch.stack(logprobs).sum().item()

        decoded = self.tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True,
        )

        optimized_prompt = decoded.split("Optimized:")[-1].strip()

        return optimized_prompt, logprob
