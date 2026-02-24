# Kymatio Modified — Learnable Wavelet Scattering

A modified version of [Kymatio](https://www.kymat.io/) that makes wavelet scattering filter parameters **learnable** via PyTorch's autograd, enabling end-to-end training of scattering-based models.

## Motivation

The standard wavelet scattering transform uses fixed, analytically designed Morlet filters. While this provides excellent mathematical guarantees (e.g., translation invariance, deformation stability), the rigid filter design may not be optimal for every task.

This modification allows the scattering filters (both the low-pass `phi` and bandpass `psi` wavelets) to be treated as **trainable `nn.Parameter` tensors**, so that gradient-based optimizers can fine-tune or fully learn them alongside a downstream classifier.

## How It Works

### Learnable filter registration

`ScatteringTorch2D` accepts a `learnable` flag. During construction the filter bank is built from Morlet wavelets in NumPy and then registered into the PyTorch module:

| `learnable` | Registration method | Gradient |
|---|---|---|
| `False` (default) | `register_buffer` | Filters are fixed tensors — no gradient |
| `True` | `register_parameter` with `nn.Parameter` | Filters participate in autograd |

The relevant code path is:

```
ScatteringTorch2D.__init__
  → ScatteringBase2D.__init__   (stores self.learnable)
  → create_filters()            (builds Morlet filter bank in NumPy)
  → register_filters()          (iterates phi/psi levels)
      → register_single_filter()
            if self.learnable:
                register_parameter(name, nn.Parameter(...))
            else:
                register_buffer(name, ...)
```

### Gradient flow through the scattering graph

All operations in the scattering pipeline are differentiable PyTorch operations:

1. **`pad`** — `ReflectionPad2d`, differentiable.
2. **`rfft` / `ifft` / `irfft`** — `torch.fft` functions, differentiable.
3. **`cdgmm`** (complex pointwise multiplication) — the filter `B` is multiplied with the signal `A` via `A * B`. Since filters are stored with a trailing real dimension (`unsqueeze(-1)`), the operation reduces to a simple element-wise multiply, which is fully differentiable with respect to the filter parameter.
4. **`subsample_fourier`** — reshape + `mean`, differentiable.
5. **`modulus`** — complex magnitude, differentiable (sub-gradient at zero).

Because every step preserves the computation graph, gradients flow from the loss, through the classifier head, back through the entire scattering transform, all the way to the filter coefficients.

## Project Structure

```
kymatio_modified/
├── model.py            # PyTorch wrapper: ScatteringBase + ExampleLinearClassifier
├── README.md
└── kymatio/            # Modified Kymatio source tree
    └── kymatio/
        ├── scattering2d/
        │   ├── frontend/
        │   │   ├── torch_frontend.py   # ScatteringTorch2D (learnable registration)
        │   │   └── base_frontend.py    # ScatteringBase2D (stores learnable flag)
        │   ├── core/
        │   │   └── scattering2d.py     # Scattering computation graph
        │   ├── backend/
        │   │   └── torch_backend.py    # cdgmm, FFT, subsample (differentiable ops)
        │   └── filter_bank.py          # Morlet wavelet generation (NumPy)
        └── ...
```

## Usage

### Basic: fixed scattering + linear classifier

```python
from model import ExampleLinearClassifier

model = ExampleLinearClassifier(
    J=3,
    shape=(32, 32),
    in_channels=3,
    num_classes=10,
    learnable=False,       # filters are fixed (default)
)

logits = model(images)     # (B, 10)
```

### Learnable scattering filters

```python
model = ExampleLinearClassifier(
    J=3,
    shape=(32, 32),
    in_channels=3,
    num_classes=10,
    learnable=True,        # filters are nn.Parameters
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

All filter tensors now appear in `model.parameters()` and receive gradients during `loss.backward()`.

### Fine-grained training control

```python
# Freeze scattering, train only the classifier head
model.freeze_scattering()
optimizer = torch.optim.Adam(model.head_parameters(), lr=1e-3)

# Later, unfreeze scattering for joint fine-tuning
model.unfreeze_scattering()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### Runtime learnable toggle

Switch filters between `nn.Parameter` and buffer mode without rebuilding the model:

```python
model.scattering.set_learnable(True)   # promote buffers → parameters
model.scattering.set_learnable(False)  # demote parameters → buffers
```

### Custom classifier head

Subclass `ScatteringBase` and implement `_build_classifier`:

```python
from model import ScatteringBase

class MyModel(ScatteringBase):
    def _build_classifier(self):
        return nn.Sequential(
            nn.Linear(self.classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes),
        )
```

## API Reference

### `ScatteringBase(J, shape, in_channels, num_classes, L=8, max_order=2, learnable=False)`

| Parameter | Type | Description |
|---|---|---|
| `J` | `int` | Log-2 of the scattering scale |
| `shape` | `(int, int)` | Spatial dimensions `(H, W)` of the input |
| `in_channels` | `int` | Number of input channels (e.g., 3 for RGB) |
| `num_classes` | `int` | Number of output classes |
| `L` | `int` | Number of wavelet orientations (default 8) |
| `max_order` | `int` | Maximum scattering order, 1 or 2 (default 2) |
| `learnable` | `bool` | If `True`, wavelet filters are trainable parameters |

**Methods:**

| Method | Description |
|---|---|
| `forward(x)` | Full forward pass: scattering → flatten → classifier |
| `extract_features(x)` | Scattering + flatten only (no classifier) |
| `head_parameters()` | Yields classifier parameters (for separate optimizer groups) |
| `freeze_scattering()` | Set `requires_grad=False` on all scattering parameters |
| `unfreeze_scattering()` | Set `requires_grad=True` on all scattering parameters |
| `freeze_classifier()` | Set `requires_grad=False` on classifier parameters |
| `unfreeze_classifier()` | Set `requires_grad=True` on classifier parameters |

### `ScatteringTorch2D.set_learnable(learnable)`

Dynamically switches all filters between `nn.Parameter` (learnable) and buffer (fixed) mode at runtime.

## Requirements

- Python >= 3.8
- PyTorch >= 1.8
- NumPy
- SciPy

## License

Kymatio is distributed under the BSD-3-Clause license. See the `kymatio/` directory for the original license and attribution.
