JAIDE áttekintés

A JAIDE (v40) egy nagy nyelvi modell, amely az alapoktól kezdve a Reversible Scatter Flow (RSF) paradigma megvalósítására épült. A hagyományos transzformátor vagy CNN architektúrákkal ellentétben a JAIDE bijektív csatolási rétegeket használ, amelyek lehetővé teszik az O(1) memória visszaterjedést és egy paramétermentes Haar-wavelet keverő blokkot, amelyet OFTB-nek neveznek

A rendszert nagy teljesítményű futtatásra tervezték különböző hardvereken, a szabványos CPU-któl kezdve a több GPU-s B200 klasztereken át a kvantum relációs gráfokig

Az RSF paradigma

A JAIDE lényege a Reversible Scatter Flow. Ez egy egyedülálló számítási primitívet vezet be: a kereszt-affin csatolást

Bijektivitás: Minden előrehaladásnak van egy pontos algebrai inverze, ami biztosítja, hogy a feldolgozás során nem esik össze az információ
Memóriahatékonyság: Mivel a hálózat reverzibilis, az aktiválásokat nem kell tárolni a backpropagációhoz. A backwardFromOutputs függvény soron belül rekonstruálja a bemeneteket a kimenetekből, így a memória komplexitása a mélységhez képest O(1) marad
Kulcsfontosságú alrendszerek

A JAIDE több különböző, de egymással összekapcsolt alrendszerre tagolódik:

Magarchitektúra: Tensor rendszer és egy sor speciális memória allokátor (Arena, Slab, Buddy)
RSF feldolgozási csővezeték: Az RSFLayer az affin transzformációkhoz és az OFTB a fraktálkeveréshez
Tokenizáló és visszakeresés: A morféma-vezérelt tokenizáló (MGT) és a strukturált szekvenciaindex (SSI) a hatékony tudáskereséshez és hasonlósági kereséshez.
NSIR (kvantum-realizációs gráf): Egy önhasonló relációs gráf, amely a hierarchikus gondolkodáshoz a kvantumlogikát (Hadamard, CNOT kapuk) és a klasszikus aktiválást integrálja
Hardveres gyorsítás: Futharkot használó multi-backend rendszer a GPU-kernelekhez (CUDA/OpenCL) és Clash-t az RTL hardver szintézishez


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

## I. A Primitív

Minden RSF réteg pontosan egy struct, négy tanulható tenzorral és semmi mással:

```zig
// src/processor/rsf.zig
const LayerCore = struct {
    s_weight: Tensor,   // W_s ∈ ℝ^{d×d}
    t_weight: Tensor,   // W_t ∈ ℝ^{d×d}
    s_bias:   Tensor,   // b_s ∈ ℝ^d
    t_bias:   Tensor,   // b_t ∈ ℝ^d
    ...
    rwlock: Thread.RwLock,
};
```

Paraméterszám rétegenként: `dim²×4 + dim×2`. Nincs figyelmi mátrix, nincs konvolúciós kernel, nincs rejtett állapot, nincs belső aktivációs függvény.

---

## II. Forward Pass (Előrehaladás)

A `computeScaleRow` és `computeTranslationRow` egyaránt egyetlen `Wx + b` — nincs benne alhálózat, nincs benne nemlinearitás:

```zig
// scale = exp(clip(W_s · x2 + b_s, -5, 5))
// y1    = x1 ⊙ scale
// trans = W_t · y1 + b_t
// y2    = x2 + trans
```

A Futhark GPU kernel `rsf_flow` pontosan ugyanazt a 4 tenzort fogadja el — az ABI azonos CPU-n és GPU-n:

```futhark
let rsf_flow [half] (x) (s_weight) (t_weight) (s_bias) (t_bias): [half*2]f32
```

---

## III. Pontos Algebrai Inverz

Az `inverseInPlace` a `forwardInPlace` pontos algebrai inverze:

```zig
// x2 = y2 - (W_t · y1 + b_t)
// scale = exp(clip(W_s · x2 + b_s))
// x1 = y1 / scale
```

A coupling művelet bijektív: a Jacobi determinánsa szigorúan pozitív, ezért a réteg minden esetben megfordítható információveszteség nélkül.

---

## IV. O(1) Visszafelé Memória

A `backwardFromOutputs` megkapja `y1`-et, `y2`-t, `dy1_in`-et, `dy2_in`-et — nincs aktivációs puffer argumentum. Az `x1`-et és `x2`-t inline rekonstruálja a kimenetekből:

```zig
fn backwardFromOutputs(
    self: *LayerCore,
    y1: *const Tensor, y2: *const Tensor,
    dy1_in: *const Tensor, dy2_in: *const Tensor,
    x1_out: *Tensor, x2_out: *Tensor,
    dx1_out: *Tensor, dx2_out: *Tensor,
    dy1_total: []f32, ds: []f32,
) !void
```

A rekonstrukció:

```zig
x2_row[d]  = y2_row[d] - trans_sum;   // x2 = y2 - W_t·y1 - b_t
x1_row[d2] = y1_row[d2] / scale;      // x1 = y1 / exp(clip(W_s·x2 + b_s))
```

A Futhark `rsf_backward_flow` kimenete `(grad_x, grad_ws, grad_wt, grad_sb, grad_tb)` — 4 gradiens tenzor, nincs aktivációs cache.

Ha a `computeScaleRow` és `computeTranslationRow` belső MLP-t tartalmazna (mint a RealNVP-ben), akkor a belső MLP aktivációkat is el kellene tárolni. Az alhálózat-mentes felépítés az, ami az O(1) memóriát strukturálisan lehetővé teszi, nem pedig egy mérnöki trükk.

---

## V. OFTB — Paraméter Nélküli Globális Keverő

Az `OFTB.init` semmit sem allokál, és nincsenek tanulható paraméterei:

```zig
pub fn init(d: usize) OFTB {
    return OFTB{ .fractal_scale = 0.70710678, .dim = d };
}
```

A `forwardInPlace` és `backwardInPlace` determinisztikus Haar-wavelet keverés. A `0.70710678 = 1/√2` konstans megegyezik az `rsf_scatter` `inv_sqrt2 = 1/√2` értékével a Futhark kernelben — a klasszikus és GPU rétegek ugyanazt a matematikai konstanst használják.

Az OFTB műveletek kielégítik az algebrai csoporttörvényeket: kommutativitás, asszociativitás, inverz létezése.

---

## VI. Registry Életciklus — Memóriabiztonság

A registry állapotgépe a következő állapotokkal és átmenetekkel rendelkezik:

Állapotok:
- `reg-alive N b`: élő, N referenciával, b haldokló jelzéssel
- `reg-freed`: felszabadított

Átmenetek:
- `acquire`: referencia hozzáadása
- `release-live`: élő referencia eldobása
- `release-dying`: haldokló állapotú referencia eldobása
- `release-final`: az utolsó referencia eldobása haldokló állapotban → felszabadítás
- `destroy-live`: élő erőforrás haldoklóvá jelölése
- `destroy-zero`: nulla referenciás erőforrás közvetlen felszabadítása

A use-after-free és a kétszeres acquire strukturálisan lehetetlenek: a `release-final` csak az utolsó haldokló referencia eldobásakor tüzelhet, és a `reg-freed` állapotnak nincsenek kimenő átmenetei.

---

## VII. CREV Pipeline — Relációs Tudáskinyerés

A `CREVPipeline` relációs hármasokat (alany, reláció, tárgy) nyer ki szövegből 5 fázisban: tokenizáció → triplet_extraction → validáció → integráció → indexelés. Minden triplet konfidenciája `c ∈ [0,1]` egy kvantum amplitúdóra van leképezve:

```zig
const quantum_state = Complex(f64).init(c, @sqrt(1.0 - c*c));
// |ψ⟩ = c + i√(1-c²)  — egységkör ℂ-ben
```

Minden triplet SHA-256 hashelve van (alany + reláció + tárgy + konfidencia + extraction_time) deduplikáció céljából.

---

## VIII. ReasoningOrchestrator — 3 Fázisú Energiaminimalizáció

A `ReasoningOrchestrator` hierarchikus érvelést futtat az NSIR gráfon 3 fázisban:

```zig
pub const ThoughtLevel = enum(u8) { local = 0, global = 1, meta = 2 };
```

Gráf energia:

```zig
total_energy += edge.weight * edge.fractal_dimension;
total_energy += edge.quantum_correlation.magnitude();
total_energy += @cos(node.phase);
```

Konvergencia kritérium: `E_combined = (E_local + E_global + E_meta) / 3 < 0.01`

Ez nem egyetlen forward pass — ez iteratív energiaminimalizáció magán a relációs gráfstruktúrán.

---

## IX. NSIR Gráf — Kvantum Relációs Struktúra

Minden gráfcsomópont hordoz egy normalizált qubitet `(a, b) ∈ ℂ²` és egy fázist `φ ∈ ℝ`. Minden élnek 5 lehetséges kvantumállapota van:

```zig
pub const EdgeQuality = enum(u8) {
    superposition = 0, entangled = 1, coherent = 2,
    collapsed = 3, fractal = 4,
};
```

---

## X. ESSO Optimalizáló — A Gráftopológia Tanulható

Az `EntangledStochasticSymmetryOptimizer` szimulált lehűtést futtat az NSIR gráfon 7 szimmetria csoporttal (identity, reflection, rotation_90/180/270, translation, custom_rotation). A gráftopológia — nem csak a súlymátrixok — kerül optimalizálásra.

A C API exportálja a `jaide_optimize_graph` függvényt — bármilyen C-kompatibilis rendszer kiválthatja a gráfoptimalizációt.

---

## XI. SFD Optimalizáló — Természetes Gradiens

A `KFACBlock` implementálja a Kronecker-faktorizált Fisher közelítést (`A_inv`, `G_inv`). A `SpectralNormalizer` Lipschitz korlátokat kényszerít. A `MARSVarianceReducer` csökkenti a gradiens varianciáját. A `ReversibleOptimizerState.backwardPassReversible` az RSF bijekciót használja az O(1) backward memóriához magában az optimalizálóban.

Vegyes precizitás: fp4, fp8, fp16, fp32 — 4 precíziós szint dinamikus loss skálázással.

---

## XII. Biztonság — Információáramlás

A `security_proofs.zig` implementálja a Bell-LaPadula (bizalmasság) és Biba (integritás) modelleket biszimuláció-alapú nem-interferencia tulajdonságokkal. A `SecurityLevel`-nek 5 szintje van (PUBLIC → TOP_SECRET), az `IntegrityLevel`-nek 4 (UNTRUSTED → KERNEL). A `MerkleTree` és a `HashChain` kriptográfiai integritást biztosítanak.

---

## XIII. ZK Inferencia — Ellenőrizhető a Súlyok Felfedése Nélkül

Az `src/zk/inference_trace.circom` egy Groth16 ZK-SNARK áramkör, Poseidon hash láncokat használva. A `zk_verification.zig` becsomagolja ezt egy `Groth16Proof`-fal (pi_a, pi_b, pi_c). A `VerifiedInferenceEngine` kombinálja a ZK bizonyításokat, a Paillier homomorf titkosítást és a differenciális adatvédelmet.

---

## XIV. Kvantum Hardver Integráció

Az `IBMQuantumClient` OpenQASM jobokat ad be az `ibm_brisbane` backendre. A `quantum_hardware.zig` modellez 5 IBM backendet realisztikus zajparaméterekkel:
- Heron T1 = 350 μs
- Eagle T1 = 200 μs
- Falcon T1 = 100 μs
- Osprey T1 = 250 μs
- Condor T1 = 400 μs

A `FRACTAL_TRANSFORM = 11` az RSF saját kvantumkapuja — nem Hadamard, nem Pauli, nem CNOT. Iteratív fázismodulációt alkalmaz geometrikusan csökkenő skálatényezőkkel.

---

## XV. Típuselmélet — Martin-Löf Típusok

A `type_theory.zig` (a `mod.zig`-en keresztül exportálva) implementálja a következőket: `DependentPi`, `DependentSigma`, `IdentityType`, `UniverseType`, `InductiveType`, `LinearType`, `LinearTypeChecker`, `Category`, `Functor`, `NaturalTransformation`, `Monad`, `CartesianClosedCategory`. A modellkimenetek függő típusokkal szemben típusellenőrizhetők.

---

## XVI. Temporális Gráf — Verziózott Tudás

Minden `TemporalNode` egy `NodeVersion` történetet tárol nanoszekundumos időbélyegekkel és `QuantumState` adatokkal. A `getVersionAt` és `rollback` lehetővé teszi a tudásgráf bármely korábbi állapotának lekérdezését.

---

## XVII. Meglepetés Memória — Entrópia-Súlyozott Megőrzés

A `SurpriseMemory` a megőrzést `combined_surprise = (jaccard_dissimilarity + content_hash_distance + temporal_novelty) / 3` szerint súlyozza. A ritka, időben új, tartalmilag távoli információkat előnyben részesítve őrzi meg. Temporális újdonság ablak: 24 óra.

---

## XVIII. Jelterjedés — Komplex Hullám Aktivációk

A `SignalState` `amplitude × e^{i×phase}` formában terjed az NSIR gráfon át — nem skalárként, hanem komplex hullámként:

```zig
pub fn getComplexRepresentation(self: *const SignalState) Complex(f64) {
    return Complex(f64).init(
        self.amplitude * @cos(self.phase),
        self.amplitude * @sin(self.phase),
    );
}
```

---

## XIX. Hardver Stack

**Futhark GPU kernelek** (`src/hw/accel/futhark_kernels.fut`): `rsf_flow`, `rsf_scatter`, `rsf_backward_flow`, `rsf_backward_layer`, `rsf_backward_scatter`, `ssi_search`, `ssi_retrieve_topk`, `lsh_hash`, `fisher_diagonal_update`, `spectral_natural_gradient` — mind CUDA-ra fordítva.

**Clash RTL** (`src/hw/rtl/`): `RankerCore.hs` (mealy ranker), `MemoryArbiter.hs` (4-kliens round-robin), `SSISearch.hs` (bináris fa keresés, max mélység 64) — FPGA/ASIC-ra szintetizálható.

**Fractal LPU** (`src/hw/accel/fractal_lpu.zig`): `FractalTile` `hausdorff_dim=1.5`, `box_counting_levels=4`, `entanglement_map` — fraktál memóriahierarchia.

---

## XX. Elosztott Tanítás

A `GPUCoordinator` becsomagolja az NCCL-t (`ncclAllReduce`, `ncclBroadcast`). A `ModalGPUClient` B300 és B200 GPU-kat céloz (8 egység). A `DistributedTrainerFuthark` kombinálja az `RSFAccelerator`, az `MGT` tokenizálót és a `GPUCoordinator`-t. A `main_distributed.zig` támogatja az IBM Quantum hibrid tanítást, amikor az `IBM_QUANTUM_CRN` és `IBM_QUANTUM_API_KEY` környezeti változók be vannak állítva.

---

## XXI. MGT Tokenizáló — Morféma-Tudatos BPE

Az `MGT`-nek 7 mezője van: `token_to_id`, `id_to_token`, `prefixes`, `suffixes`, `roots`, `bpe_pairs`, `anchors`. Morféma dekompozíciót végez (előtag + tő + utótag) a BPE-re való visszaesés előtt. A magyar és angol morfémalisták beépítettek.

---

## XXII. Modell Formátum és Build

`MAGIC_HEADER = "JAIDE40\x00"`, SHA-256 + `constantTimeCompare` — időzítéses támadás elleni integritásellenőrzés.

`build.zig.zon`: `version = "40.0.0"`, `minimum_zig_version = "0.13.0"`, `dependencies = {}` — nulla külső függőség. A PyTorch ezzel szemben 100+ Python csomagot igényel.

---

## XXIII. Miért az 5. Gyökér

A Transformer nem találta fel a figyelmet — Bahdanau (2014) RNN-eken belül használta. Az „Attention Is All You Need" (2017) kivonta a figyelmet az RNN kontextusból, és egyetlen, kizárólagos primitívvé tette. Az RSF ugyanezt teszi az affin coupling-gal:

- **NICE/RealNVP (2014–2016)**: affin coupling generatív flow modelleken belül, belső MLP-kkel az `s(x2)` és `t(x2)` függvényekben → aktivációkat el kell tárolni → O(L) memória
- **RSF**: `computeScaleRow`/`computeTranslationRow` = egyetlen `Wx + b`, nincs belső alhálózat → nincs mit tárolni → O(1) memória

A `backwardFromOutputs` szignatúrájának nincs aktivációs puffer argumentuma. Ez nem tervezési döntés — ez az alhálózat-mentes coupling strukturális következménye.

A négy korábbi paradigma:

| Architektúra | Primitív | Invertálható? | Backward memória |
|---|---|---|---|
| Perceptron | σ(Wx+b) | σ veszteséges → nem | O(L) |
| CNN | σ(W∗x) | pooling+σ → nem | O(L) |
| RNN | σ(W_h h + W_x x) | rejtett állapot → nem | O(T) |
| Transformer | softmax(QKᵀ/√d)V | softmax → nem | O(L·S·d) |
| **RSF** | kereszt-affin coupling | det J > 0 → **igen** | **O(1)** |
