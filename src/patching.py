"""Activation patching of the anchor column (all depths).

Used by the toy gate (swap/override controls) and by Exp 2 conditions
anchor_shuffled and anchor_mean. The capsule is the anchor's hidden-state
column at every depth (embedding output + each decoder layer output, see
docs/mechanism.md §3); patching replaces that whole column.
"""

from __future__ import annotations

import torch


def decoder_layers(model) -> torch.nn.ModuleList:
    lists = [m for _, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) >= 8]
    if not lists:
        raise RuntimeError("could not locate decoder layers")
    return max(lists, key=len)


class AnchorPatcher:
    """Capture the anchor hidden state at every depth from one forward, then
    force those states into subsequent forwards (activation patching)."""

    def __init__(self, model, pos: int):
        self.modules = [model.get_input_embeddings(), *list(decoder_layers(model))]
        self.pos = pos
        self.stored: list[torch.Tensor | None] = [None] * len(self.modules)
        self._handles: list = []

    @staticmethod
    def _tensor(output):
        return output[0] if isinstance(output, tuple) else output

    def capture(self, run) -> None:
        def make(i):
            def hook(_m, _args, output):
                self.stored[i] = self._tensor(output)[:, self.pos, :].detach().clone()
            return hook
        handles = [m.register_forward_hook(make(i)) for i, m in enumerate(self.modules)]
        try:
            run()
        finally:
            for h in handles:
                h.remove()

    def capture_context_mean(self, run, context_end: int,
                             norm_match: bool = True) -> None:
        """Store the MEAN of context hidden states (positions < context_end)
        at every depth, optionally rescaled to the anchor's own norm at that
        depth (Exp 2 'anchor_mean' diagnostic: structured noise control)."""
        def make(i):
            def hook(_m, _args, output):
                t = self._tensor(output)
                mean_vec = t[:, :context_end, :].mean(dim=1)
                if norm_match:
                    anchor = t[:, self.pos, :]
                    scale = anchor.norm(dim=-1, keepdim=True) / \
                        mean_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    mean_vec = mean_vec * scale
                self.stored[i] = mean_vec.detach().clone()
            return hook
        handles = [m.register_forward_hook(make(i)) for i, m in enumerate(self.modules)]
        try:
            run()
        finally:
            for h in handles:
                h.remove()

    def retarget(self, pos: int) -> None:
        """Patch the stored states at a different anchor position (donor and
        host prompts may place [COMPRESS] at different indices)."""
        self.pos = pos

    def __enter__(self):
        def make(i):
            def hook(_m, _args, output):
                t = self._tensor(output)
                t[:, self.pos, :] = self.stored[i].to(t.dtype)
                return output
            return hook
        self._handles = [m.register_forward_hook(make(i))
                         for i, m in enumerate(self.modules)]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False
