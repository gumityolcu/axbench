from pathlib import Path

import torch

from .model import Model
from axbench.utils.model_utils import gather_residual_activations


class VanillaVAE(Model):
    def __str__(self):
        return "VanillaVAE"

    def make_model(self, **kwargs):
        self.concept_id = kwargs.get("concept_id", None)
        self.dump_dir = kwargs.get("dump_dir", self.dump_dir)
        self.metadata_path = kwargs.get("metadata_path", None)
        self.lm_model_name = kwargs.get("lm_model_name", None)
        self.last_train_result = None
        self.vae = None
        self.checkpoint = None

    def train(self, examples, **kwargs):
        self._train(examples, auxiliary_examples=None, **kwargs)

    def _train(
        self,
        examples,
        auxiliary_examples=None,
        auxiliary_prefix_length=1,
        **kwargs,
    ):
        if getattr(self, "concept_id", None) is None:
            self.concept_id = kwargs.get("concept_id", None)
        if self.concept_id is None:
            logging_metadata = kwargs.get("logging_metadata", {})
            self.concept_id = logging_metadata.get("concept_id", None)
        if self.concept_id is None:
            raise ValueError(f"{self} requires concept_id during training")
        metadata_path = kwargs.get("metadata_path", getattr(self, "metadata_path", None))
        if metadata_path is None:
            raise ValueError(f"{self} requires metadata_path during training")
        if getattr(self, "dump_dir", None) is None:
            raise ValueError(f"{self} requires dump_dir during training")

        from axbench.geocanon.vanilla_vae import train_vanilla_vae

        self.last_train_result = train_vanilla_vae(
            model=self.model,
            tokenizer=self.tokenizer,
            examples=examples,
            metadata_path=metadata_path,
            concept_id=int(self.concept_id),
            layer=self.layer,
            dump_dir=self.dump_dir,
            training_args=self.training_args,
            prefix_length=kwargs.get("prefix_length", 1),
            device=self.device,
            lm_model_name=getattr(self, "lm_model_name", None),
            auxiliary_examples=auxiliary_examples,
            auxiliary_prefix_length=auxiliary_prefix_length,
            model_name=self.__str__(),
        )

    def save(self, dump_dir, **kwargs):
        if self.last_train_result is None:
            return
        model_name = kwargs.get("model_name", self.__str__())
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)

        weight_file = dump_dir / f"{model_name}_weight.pt"
        weight = self.last_train_result["classifier_weight"].cpu()
        if weight_file.exists():
            weight = torch.cat([torch.load(weight_file, map_location="cpu"), weight], dim=0)
        torch.save(weight, weight_file)

        bias_file = dump_dir / f"{model_name}_bias.pt"
        bias = self.last_train_result["classifier_bias"].cpu()
        if bias_file.exists():
            bias = torch.cat([torch.load(bias_file, map_location="cpu"), bias], dim=0)
        torch.save(bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        concept_id = kwargs.get("concept_id", getattr(self, "concept_id", None))
        if concept_id is None:
            raise ValueError(f"{self} requires concept_id when loading")
        self.concept_id = int(concept_id)
        self.dump_dir = dump_dir or getattr(self, "dump_dir", None)
        if self.dump_dir is None:
            raise ValueError(f"{self} requires dump_dir when loading")
        from axbench.geocanon.vanilla_vae import load_vanilla_vae

        self.vae, self.checkpoint = load_vanilla_vae(
            self.dump_dir,
            self.concept_id,
            self.device,
            model_name=self.__str__(),
        )

    def to(self, device):
        self.device = device
        if getattr(self, "vae", None) is not None:
            self.vae.to(device)
        return self

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        if getattr(self, "vae", None) is None:
            concept_id = kwargs.get("concept_id", getattr(self, "concept_id", None))
            if concept_id is None and "concept_id" in examples:
                concept_id = int(examples["concept_id"].iloc[0])
            self.load(dump_dir=self.dump_dir, concept_id=concept_id)

        batch_size = kwargs.get("batch_size", 32)
        prefix_length = kwargs.get("prefix_length", 1)
        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []

        for start in range(0, len(examples), batch_size):
            batch = examples.iloc[start:start + batch_size]
            inputs = self.tokenizer(
                batch["input"].tolist(),
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)
            activations = gather_residual_activations(self.model, self.layer, inputs).detach()
            token_activations = activations[:, prefix_length:].float()
            batch_shape = token_activations.shape
            flat_activations = token_activations.reshape(-1, batch_shape[-1])
            z_mean = self.vae.encoder(flat_activations).base_dist.loc
            logits = self.vae.classifier(z_mean).reshape(batch_shape[0], batch_shape[1])
            seq_lens = inputs["attention_mask"].sum(dim=1) - prefix_length

            for seq_idx in range(batch_shape[0]):
                length = int(seq_lens[seq_idx].item())
                scores = logits[seq_idx, :length].float().cpu().tolist()
                scores = [round(float(score), 3) for score in scores]
                max_act = max(scores) if scores else 0.0
                max_idx = scores.index(max_act) if scores else 0
                tokens = self.tokenizer.tokenize(batch.iloc[seq_idx]["input"])[prefix_length - 1:]
                max_token = tokens[max_idx] if max_idx < len(tokens) else ""
                all_acts.append(scores)
                all_max_act.append(max_act)
                all_max_act_idx.append(max_idx)
                all_max_token.append(max_token)
                all_tokens.append(tokens)

            del activations
            torch.cuda.empty_cache()

        return {
            "acts": all_acts,
            "max_act": all_max_act,
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token,
            "tokens": all_tokens,
        }

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        concept_id = kwargs.get("concept_id", getattr(self, "concept_id", None))
        if concept_id is None:
            raise ValueError(f"{self} requires concept_id for steering")
        if isinstance(concept_id, list):
            concept_id = concept_id[0]
        if getattr(self, "vae", None) is None or self.concept_id != int(concept_id):
            self.load(dump_dir=self.dump_dir, concept_id=int(concept_id))

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        batch_size = kwargs.get("batch_size", 16)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        prefix_length = kwargs.get("prefix_length", 1)
        use_synergy = kwargs.get("use_synergy", False)
        intervene_on_prompt = kwargs.get("intervene_on_prompt", True)
        all_generations = []
        all_strengths = []

        classifier_direction = self.vae.classifier.weight.detach().flatten()
        direction_norm = classifier_direction.norm().clamp_min(torch.finfo(classifier_direction.dtype).eps)
        classifier_direction = classifier_direction / direction_norm

        modules = self.model.model.layers

        try:
            for start in range(0, len(examples), batch_size):
                batch = examples.iloc[start:start + batch_size]
                input_strings = (
                    batch["steered_input"].tolist()
                    if use_synergy and "steered_input" in batch
                    else batch["input"].tolist()
                )
                factors = torch.tensor(
                    batch["factor"].tolist(),
                    device=self.device,
                    dtype=classifier_direction.dtype,
                )
                inputs = self.tokenizer(
                    input_strings,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(self.device)
                attention_mask = inputs["attention_mask"].bool()

                def hook(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    original_dtype = hidden.dtype
                    hidden_float = hidden.float()
                    batch_n, seq_len, hidden_dim = hidden_float.shape
                    active_mask = torch.ones(
                        (batch_n, seq_len),
                        device=hidden_float.device,
                        dtype=torch.bool,
                    )
                    if seq_len == attention_mask.shape[1]:
                        active_mask = attention_mask[:, -seq_len:].clone()
                        if not intervene_on_prompt:
                            active_mask[:] = False
                        else:
                            for row_idx in range(batch_n):
                                valid_positions = active_mask[row_idx].nonzero(as_tuple=False).flatten()
                                active_mask[row_idx, valid_positions[:prefix_length]] = False

                    flat_hidden = hidden_float.reshape(-1, hidden_dim)
                    q = self.vae.encoder(flat_hidden)
                    z_mean = q.base_dist.loc
                    z_shift = factors[:, None, None].expand(batch_n, seq_len, -1)
                    z_shift = z_shift.reshape(-1, 1) * classifier_direction.reshape(1, -1)
                    decoded = self.vae.decoder(z_mean + z_shift).mean
                    baseline = self.vae.decoder(z_mean).mean
                    latent_delta = (decoded - baseline).reshape(batch_n, seq_len, hidden_dim)
                    active_mask_3d = active_mask.unsqueeze(-1)
                    steered = torch.where(
                        active_mask_3d,
                        hidden_float + latent_delta,
                        hidden_float,
                    ).to(original_dtype)
                    if isinstance(output, tuple):
                        return (steered,) + output[1:]
                    return steered

                handle = modules[self.layer].register_forward_hook(hook, always_call=True)
                try:
                    generations = self.model.generate(
                        **inputs,
                        max_new_tokens=eval_output_length,
                        do_sample=True,
                        temperature=temperature,
                    )
                finally:
                    handle.remove()

                input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
                generated_texts = [
                    self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                    for generation, input_length in zip(generations, input_lengths)
                ]
                all_generations.extend(generated_texts)
                all_strengths.extend(factors.detach().float().cpu().tolist())
                torch.cuda.empty_cache()
        finally:
            self.tokenizer.padding_side = old_padding_side

        return {
            "steered_generation": all_generations,
            "strength": all_strengths,
        }


class AuxiliaryVAE(VanillaVAE):
    """Classified VAE whose reconstruction objective uses a separate text corpus."""

    def __str__(self):
        return "AuxiliaryVAE"

    def train(self, examples, **kwargs):
        from axbench.geocanon.vanilla_vae import load_auxiliary_texts

        source = kwargs.pop("auxiliary_examples", None)
        if source is None:
            source = getattr(self.training_args, "vae_auxiliary_data_path", None)
        if source is None:
            raise ValueError(
                "AuxiliaryVAE requires auxiliary_examples or vae_auxiliary_data_path"
            )
        text_column = getattr(self.training_args, "vae_auxiliary_text_column", None) or "text"
        max_examples = getattr(self.training_args, "vae_auxiliary_max_examples", None)
        auxiliary_prefix_length = getattr(
            self.training_args,
            "vae_auxiliary_prefix_length",
            None,
        )
        if auxiliary_prefix_length is None:
            auxiliary_prefix_length = 1
        auxiliary_examples = load_auxiliary_texts(
            source,
            text_column=text_column,
            max_examples=max_examples,
        )
        self._train(
            examples,
            auxiliary_examples=auxiliary_examples,
            auxiliary_prefix_length=auxiliary_prefix_length,
            **kwargs,
        )
