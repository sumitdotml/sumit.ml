import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import marimo as mo
    return F, mo, nn, torch


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
def _(torch):
    def renormalization(input: torch.Tensor) -> torch.Tensor:
        assert type(input) is torch.Tensor, "input must be a torch.tensor"
        if input.dim() != 1:
            input.squeeze()
        renormalized = torch.zeros_like(input)
        total = torch.sum(input)
        for i, score in enumerate(input):
            renormalized[i] = score / total
        return renormalized    
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
def _(F, nn, torch):
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    residual_stream = torch.randn(4, 16)
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
def _(scores, torch):
    def softmax(input: torch.Tensor) -> torch.Tensor:
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
def _(nn, torch):
    class GLU(nn.Module):
        def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
            super.__init__()
            self.W_gate = None
            self.W_up = None
            self.W_down = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pass
    return


@app.cell
def _():
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
    - n_topk = 2
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. raw score computation for all experts
    """)
    return


if __name__ == "__main__":
    app.run()
