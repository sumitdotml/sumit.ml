import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    return (torch,)


@app.cell
def _(torch):
    a = 0.7
    b = 0.6

    to_normalize = torch.tensor([a, b])
    return a, b, to_normalize


@app.cell
def _(to_normalize, torch):
    normalized = torch.softmax(to_normalize, dim=-1)
    normalized
    return


@app.cell
def _(a, b):
    norm_a = a / (a + b)
    norm_b = b / (a + b)
    norm_a, norm_b
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
