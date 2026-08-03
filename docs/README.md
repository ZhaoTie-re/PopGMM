# PopGMM documentation

Start at [`../README.md`](../README.md) if you want to install the pipeline, run
it, and use the sample lists it produces. These documents cover everything else.

| Document | Answers | For |
|---|---|---|
| [`method.md`](method.md) | Why each of the eight stages exists, what it computes, and the limitations of the whole approach | Reviewers, and anyone deciding whether the method fits their study |
| [`outputs.md`](outputs.md) | What every file under `results/` contains, and how to check a run | Anyone reading the results or reproducing them |
| [`gmm_convergence_diagram.EN.md`](gmm_convergence_diagram.EN.md) | The EM iteration, convergence criteria, BIC, and the Mahalanobis component distance, derived | Anyone verifying the mathematics |
| [`gmm_convergence_diagram.CN.md`](gmm_convergence_diagram.CN.md) | 同上,中文版 | — |
| [`reproducibility_probe.md`](reproducibility_probe.md) | Whether the pipeline is bit-deterministic, and what the float32 → float64 change fixed | Anyone who gets a different number than the committed one |
| [`environment_snapshot.txt`](environment_snapshot.txt) | The exact interpreter, BLAS build and package versions behind the committed artifacts | Anyone reproducing them |

`published_structure.png` is the reference figure embedded in
[`method.md`](method.md#4-merging-components-into-ancestry-regions); it is not a
document on its own.
