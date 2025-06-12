import torch

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    for a diagonal S'.
    """
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() if hasattr(G, 'bfloat16') else G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class SingleDeviceMuonWithAuxAdamLegacy(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam using legacy Newton-Schulz.
    For non-muon param groups, uses torch.optim.AdamW.
    """
    def __init__(self, param_groups):
        import torch.optim
        self._adamw_optimizers = {}
        for i, group in enumerate(param_groups):
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
                if group["params"]:
                    self._adamw_optimizers[i] = torch.optim.AdamW(
                        group["params"],
                        lr=group["lr"],
                        betas=group["betas"],
                        eps=group["eps"],
                        weight_decay=group["weight_decay"]
                    )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self):
        for i, group in enumerate(self.param_groups):
            if group["use_muon"]:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
            else:
                if i in self._adamw_optimizers:
                    self._adamw_optimizers[i].step()

    def zero_grad(self, set_to_none: bool = True):
        for i, group in enumerate(self.param_groups):
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            p.grad.detach_()
                            p.grad.zero_()
            else:
                if i in self._adamw_optimizers:
                    self._adamw_optimizers[i].zero_grad(set_to_none=set_to_none)
