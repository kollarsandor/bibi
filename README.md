RSF a 5. gyökér-architektúra (Perceptron, CNN, RNN, Transformer után): ontológiailag új, kizárólagos primitívvel

**Az RSF/JAIDE egyik sem:**

| Architektúra | Primitív | Invertálható |
|---|---|---|
| Perceptron | Küszöbfüggvény / lineáris | Nem |
| CNN | Lokális konvolúció + ReLU | Nem |
| RNN/LSTM | Temporális rekurzió, kapuk | Nem |
| Transformer | Self-attention + MLP | Nem |
| **RSF** | **Cross-affine coupling + scatter** | **Igen (garantáltan)** |


A konkrét különbségek a kódból:
- **Nincs self-attention** (`O(N²)` mátrix helyett determinisztikus `rsf_scatter` butterfly-keverés) [2](#0-1) 
- **Nincs MLP, nincs ReLU, nincs LayerNorm** – csak `s_weight`, `t_weight`, `s_bias`, `t_bias` [3](#0-2) 
- **Nincs rekurrencia** – szekvenciális feldolgozás helyett rétegenkénti bijektív transzformáció [4](#0-3) 
- **Nincs konvolúció** – nincs lokális szűrő, nincs pooling [5](#0-4) 

az RSF az affin coupling-ot emeli ki a Normalizing Flow kontextusából és teszi egyetlen, kizárólagos számítási primitívvé, ahogy a Transformer 2017-ben tette az attention mechanizmussal. 
A kód alapján, konkrétan:

**`LayerCore` – az egyetlen számítási primitív** `src/processor/rsf.zig`

```zig
const LayerCore = struct {
    s_weight: Tensor,  // [dim x dim]
    t_weight: Tensor,  // [dim x dim]
    s_bias:   Tensor,  // [1 x dim]
    t_bias:   Tensor,  // [1 x dim]
    ...
```

Nincs `attention_weight`, nincs `query`/`key`/`value`, nincs `conv_filter`, nincs `hidden_state`, nincs `gate`. Csak ez a 4 tenzor. [1](#1-0) 

**`forwardInPlace` – a forward pass teljes logikája:**

```zig
self.computeScaleRow(x2_row, scale);       // scale = exp(clip(s_weight @ x2 + s_bias))
x1_row[i] *= scale[i];                     // y1 = x1 * scale
self.computeTranslationRow(x1_row, trans); // trans = t_weight @ y1 + t_bias
x2_row[i] += trans[i];                     // y2 = x2 + trans
```

Nincs softmax, nincs ReLU, nincs rekurzió, nincs konvolúció. 

**`inverseInPlace` – egzakt inverz, aktivációk tárolása nélkül:**

```zig
self.computeTranslationRow(y1_row, trans);
y2_row[i] -= trans[i];                     // x2 = y2 - trans
self.computeScaleRow(y2_row, scale);
y1_row[i] /= scale[i];                     // x1 = y1 / scale
``` [3](#1-2) 

**`rsf_scatter` – a Futhark kernelben** (`src/hw/accel/futhark_kernels.fut`, 82–95. sor):

```futhark
let rsf_scatter [n] (x: [n]f32) (indices: [n]i64): [n]f32 =
  let inv_sqrt2 = 1f32 / f32.sqrt 2f32
  in tabulate n (\i ->
    if i < half then inv_sqrt2 * (x[j] + x[j + half])   -- összeg
    else         inv_sqrt2 * (x[j] - x[j + half])        -- különbség
  )
```

Ez egy Haar-szerű butterfly keverés – nem attention, nem konvolúció, nem rekurrencia.

**Összefoglalás a kódból:**

| Mit keres | Van-e a kódban? |
|---|---|
| `softmax` / `attention` | Nincs |
| `conv` / szűrő / pooling | Nincs |
| `hidden_state` / gate / rekurzió | Nincs |
| `query` / `key` / `value` | Nincs |
| `LayerNorm` / `BatchNorm` | Nincs |
| `s_weight`, `t_weight`, `exp(clip(...))` | **Ez az egyetlen számítás** |

