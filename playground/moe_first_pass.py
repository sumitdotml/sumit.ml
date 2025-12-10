import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import marimo as mo
    from typing import Optional
    from torch import Tensor
    return F, Optional, Tensor, mo, nn, torch


@app.cell
def _(torch):
    torch.manual_seed(40)

    router_scores = torch.rand(8)
    router_scores
    return (router_scores,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Softmax
    """)
    return


@app.cell
def _(router_scores):
    softmaxed = router_scores.softmax(dim=-1)
    softmaxed
    return (softmaxed,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Topk
    """)
    return


@app.cell
def _(softmaxed, torch):
    topk = torch.topk(softmaxed, 2).values
    topk
    return (topk,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Renormalization (not softmax)
    """)
    return


@app.cell
def _(Tensor, torch):
    def renormalization(input: Tensor) -> Tensor:
        assert type(input) is Tensor, "input must be a torch.tensor"
        if input.dim() != 1:
            input.squeeze()
        renormalized = torch.zeros_like(input)
        total = torch.sum(input)
        for i, score in enumerate(input):
            renormalized[i] = score / total
        return renormalized

    # UPDATE: this renormalization is wrong. see the way I've done it below in MoERouter
    return (renormalization,)


@app.cell
def _(renormalization, topk):
    renormalized_topk = renormalization(topk)
    renormalized_topk
    return (renormalized_topk,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Weighted Combination
    """)
    return


@app.cell
def _(F, Tensor, nn):
    class FFN_Expert(nn.Module):
        """
        SwiGLU feed-forward module.

        implements the SwiGLU variant of the GLU activation function
        with a pair of projection matrices and SiLU (Swish) activation.
        """

        def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
            """
            SwiGLU module initialization

            Args:
                hidden_dim (int): Input and output dimension
                ffn_dim (int): Intermediate dimension for the feed-forward network
            """
            super().__init__()
            self.hidden_dim = hidden_dim
            self.ffn_dim = ffn_dim

            self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            """
            Forward pass for the SwiGLU module.

            Args:
                x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_dim]

            Returns:
                torch.Tensor: Output tensor [batch_size, seq_len, hidden_dim]
            """
            # Gate and up projections
            gate = self.w_gate(x)  # [batch_size, seq_len, ffn_dim]
            up = self.w_up(x)  # [batch_size, seq_len, ffn_dim]

            # Applying SiLU (Swish) activation to the gate
            activated_gate = F.silu(gate)  # [batch_size, seq_len, ffn_dim]

            # Element-wise multiplication
            intermediate = activated_gate * up  # [batch_size, seq_len, ffn_dim]

            # Down projection
            output = self.w_down(intermediate)  # [batch_size, seq_len, hidden_dim]

            return output        
    return (FFN_Expert,)


@app.cell
def _(FFN_Expert):
    ffn_expert = FFN_Expert(ffn_dim=8, hidden_dim=16)
    return (ffn_expert,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Weighted combination

    assuming my topk = 2 for this
    """)
    return


@app.cell
def _(ffn_expert, renormalized_topk, torch):
    torch.manual_seed(40)
    residual_stream = torch.randn(2, 4, 16)
    weighted_combination = renormalized_topk[0] * ffn_expert(residual_stream) + renormalized_topk[1] * ffn_expert(residual_stream)
    return (weighted_combination,)


@app.cell
def _(weighted_combination):
    weighted_combination.shape
    return


@app.cell
def _(weighted_combination):
    weighted_combination
    return


@app.cell
def _(torch):
    torch.manual_seed(40)
    scores = torch.rand(8)
    scores, sum(scores)
    return (scores,)


@app.cell
def _(Tensor, scores, torch):
    def softmax(input: Tensor) -> Tensor:
        if input.dim() != 1:
            input.squeeze()
        probability = torch.zeros_like(input)
        total = sum([torch.exp(input[j]) for j in range(len(input))])
        for i, score in enumerate(input):
            probability[i] = torch.exp(score) / total
        return probability
    print(f"""My vanilla softmax:
    {softmax(scores)}

    PyTorch's softmax:
    {torch.softmax(scores, dim=-1)}
    """)
    return


@app.cell
def _(scores, torch):
    sum([torch.exp(scores[j]) for j in range(len(scores))])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Some SwiGLU practice
    """)
    return


@app.cell
def _(router_scores):
    router_scores
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### GLU

    ```
    h = σ(W_gate · x) ⊙ (W_up · x)
    output = W_down · h
    ```

    where:
    - ⊙ = element-wise multiplication
    - σ = sigmoid function (outputs 0 to 1, the "gate")
    - W_gate · x = gating_pathway (just a vasiable holding the gating values, controls what gets through)
    - W_up · x = signal_pathway (upward projection, ups the dimension of x, carries the actual information)
    - h = hidden = sigmoud (gating_pathway) ⊙ signal_pathway
    - output = projection back to original dimension with W_out
    """)
    return


@app.cell
def _(Tensor, nn, torch):
    class GLU(nn.Module):
        def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
            super().__init__()
            self.W_gate = nn.Linear(in_features=hidden_dim, out_features=ffn_dim, bias=False)
            self.W_up = nn.Linear(in_features=hidden_dim, out_features=ffn_dim, bias=False)
            self.W_down = nn.Linear(in_features=ffn_dim, out_features=hidden_dim, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            gating_pathway = self.W_gate(x)
            signal_pathway = self.W_up(x)
            hidden = signal_pathway * torch.sigmoid(gating_pathway)
            output = self.W_down(hidden)
            return output
    return (GLU,)


@app.cell
def _(GLU):
    glu = GLU(hidden_dim=8, ffn_dim=16)
    return (glu,)


@app.cell
def _(torch):
    torch.manual_seed(41)
    x = torch.rand(2, 3, 8)
    x.shape
    return (x,)


@app.cell
def _(glu, x):
    print(glu(x))
    return


@app.cell
def _(Optional, Tensor, nn, torch):
    class GLU_Torch(nn.Module):
        r"""
        The same GLU, this time using PyTorch's own `torch.nn.functional.glu` for reference.
        """

        def __init__(
            self,
            hidden_dim: int,
            ffn_dim: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
        ) -> None:
            super().__init__()

            # mimicking how the pytorch code does this
            factory_kwargs = {"device": device, "dtype": dtype}

            self.W_gate = nn.Linear(
                in_features=hidden_dim, out_features=ffn_dim, bias=False, **factory_kwargs
            )
            self.W_up = nn.Linear(
                in_features=hidden_dim, out_features=ffn_dim, bias=False, **factory_kwargs
            )
            self.W_down = nn.Linear(
                in_features=ffn_dim, out_features=hidden_dim, bias=False, **factory_kwargs
            )

        def forward(self, x: Tensor) -> Tensor:
            gating_pathway = self.W_gate(x)
            signal_pathway = self.W_up(x)
            hidden = GLU_Torch._concatenate_and_get_glu(gating_pathway, signal_pathway)
            output = self.W_down(hidden)
            return output

        @staticmethod
        def _concatenate_and_get_glu(gate: Tensor, signal: Tensor) -> Tensor:
            return nn.functional.glu(torch.concat([signal, gate], dim=-1))
    return (GLU_Torch,)


@app.cell
def _(GLU_Torch, torch):
    torch.manual_seed(41)
    glu_torch = GLU_Torch(hidden_dim=8, ffn_dim=16)
    return (glu_torch,)


@app.cell
def _(glu_torch, x):
    print(glu_torch(x))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## SwiGLU now
    """)
    return


@app.cell
def _(Optional, Tensor, nn, torch):
    class SwiGLU(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            ffn_dim: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
        ):
            super().__init__()

            factory_kwargs = {"device": device, "dtype": dtype}

            self.W_gate = nn.Linear(
                in_features=hidden_dim, out_features=ffn_dim, **factory_kwargs
            )
            self.W_up = nn.Linear(
                in_features=hidden_dim, out_features=ffn_dim, **factory_kwargs
            )

            self.W_down = nn.Linear(
                in_features=ffn_dim, out_features=hidden_dim, **factory_kwargs
            )

        def forward(self, x: Tensor) -> Tensor:
            gating_pathway = self.W_gate(x)
            signal_pathway = self.W_up(x)
            swish = gating_pathway * torch.sigmoid(gating_pathway)
            hidden = swish * signal_pathway
            output = self.W_down(hidden)
            return output
    return (SwiGLU,)


@app.cell
def _(SwiGLU):
    swiglu = SwiGLU(hidden_dim=8, ffn_dim=16)
    return (swiglu,)


@app.cell
def _(x):
    x
    return


@app.cell
def _(swiglu, torch, x):
    torch.manual_seed(41)
    print(swiglu(x))
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Now the Router Math (Mixtral style)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Assumptions

    - experts = 8
    - residual stream input `x` dimension: `[4096]`
    - batches = 2
    - n_topk = 2
    """)
    return


@app.cell
def _(Optional, Tensor, nn, torch):
    class MoERouter(nn.Module):
        def __init__(self, hidden_dim: int, n_ffn_experts: int, topk: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
            super().__init__()
            factory_kwargs = {"device": device, "dtype": dtype}
            self.n_ffn_experts = n_ffn_experts
            self.topk = topk
            self.W_router = nn.Linear(in_features=hidden_dim, out_features=n_ffn_experts, bias=False, **factory_kwargs)

        # x is [batch*seqlen, hidden_dim]
        def forward(self, flattened_x: Tensor) -> tuple[Tensor, Tensor]:

            # Step 1: sequence dimensions
            _, hidden_dim = flattened_x.shape

            # Step 2: Compute raw scores for all experts
            router_matrix = self.W_router(flattened_x) # [batch*seqlen, n_ffn_experts]

            # Step 3: Apply softmax
            probabilities = torch.softmax(router_matrix, dim=1)

            # Step 4: Select top-k experts
            # values: [batch*seqlen, topk]
            # indices: [batch*seqlen, topk]
            values, indices = torch.topk(probabilities, k=self.topk)

            # Step 5: Renormalize weights
            values = MoERouter._renormalization(values)

            return values, indices


        @staticmethod
        def _renormalization(input: Tensor) -> Tensor:
            total = input.sum(dim=-1, keepdim=True) # total sum of experts' raw scores, not token count, hence -1 dim
            renormalized = input / total # [batch*seqlen, topk]
            return renormalized

    torch.manual_seed(40)
    x_in = torch.rand(2, 3, 8)
    moe = MoERouter(hidden_dim=8, n_ffn_experts=8, topk=2)
    values, indices = moe(x_in.view(-1, 8))
    print(f"{values.shape}\n{values}\n{indices.shape}\n{indices}")
    return MoERouter, indices, values, x_in


@app.cell
def _(SwiGLU, nn):
    experts = nn.ModuleList([SwiGLU(hidden_dim=8, ffn_dim=16) for _ in range(8)])
    return (experts,)


@app.cell
def _(experts):
    experts
    return


@app.cell
def _(MoERouter, Optional, SwiGLU, Tensor, nn, torch):
    class MoELayer(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            ffn_dim: int,
            n_experts: int,
            topk: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
        ) -> None:
            super().__init__()
            factory_kwargs = {"device": device, "dtype": dtype}

            self.hidden_dim = hidden_dim
            self.n_experts = n_experts
            self.topk = topk

            self.router = MoERouter(
                hidden_dim=hidden_dim, n_ffn_experts=n_experts, topk=topk, **factory_kwargs
            )
            self.experts = nn.ModuleList(
                [
                    SwiGLU(hidden_dim=hidden_dim, ffn_dim=ffn_dim, **factory_kwargs)
                    for _ in range(n_experts)
                ]
            )

        def forward(self, x: Tensor) -> Tensor:
            _, _, hidden_dim = x.shape
            x_flattened = x.view(-1, hidden_dim)
            weights, indices = self.router(x_flattened)
            results = torch.zeros_like(x_flattened)

            # this works, but it processes tokens sequentially, so very inefficient for production
            # but leaving it here for reference since this is the logic
            # for i, token in enumerate(flattened_x):
            #     for j in range(self.topk):
            #         results[i] += (self.experts[indices[i][j]](x_flattened[i])) * weights[i][j]
            # return results

            # fast & efficient vectorized approach
            # reference: https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/moe.py#L16
            for i, expert in enumerate(self.experts):
                token_idx, topk_idx = torch.where(indices == i)

                if len(token_idx) == 0:
                    continue

                # Run expert and accumulate weighted results
                # unsqueeze is to add 1 dim at the end since weights[token_idx, topk_idx] gives us 1d-tensors
                results[token_idx] += weights[token_idx, topk_idx].unsqueeze(-1) * expert(x_flattened[token_idx])
            return results
    return (MoELayer,)


@app.cell
def _(MoELayer, x_in):
    moelayer = MoELayer(hidden_dim=8, ffn_dim=16, n_experts=8, topk=2)
    out = moelayer(x_in) # x_in: [2, 3, 8]
    print(out)
    return


@app.cell
def _(torch):
    ind = torch.tensor([[6, 5],
            [6, 7],
            [7, 3],
            [3, 1],
            [7, 6],
            [2, 7]])

    ind
    return


@app.cell
def _(indices, torch, values):
    for i in range(8):
        a, b = torch.where(indices == i)
        if len(a) == 0:
            continue
        print(values[a, b].unsqueeze(-1).dim())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
