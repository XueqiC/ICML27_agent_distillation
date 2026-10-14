# Remark 9.6 · Contribution units and nearest prior work

*Statement*

[← Framework](../README.md) · [Statement](09-06_contribution_units_prior_work.md) · [中文](09-06_contribution_units_prior_work_ZH.md)

**In one paragraph.** Almost every tool in this framework is classical: operator approximation, predictive-state realization, performance-difference identities, two-point lower bounds, optimal design, convex realizability, control variates. A combination of classical tools can still be an original contribution, but only if it poses a problem the existing frameworks do not handle well and yields conclusions that change what a method or an experiment does. This remark states the framework's contribution in three units, each explicitly a composition claim and not new mathematics. It then lists, for each nearest line of prior work, what must be conceded and what increment the framework must still show. It does not claim that originality is established, and it says plainly what is missing.

**Setting used on this page.**

- *Contribution unit.* A claim that a reviewer can compare with prior work, refute or verify. It names the problem, the composed result and the decision it changes.
- *Composition claim.* A claim that the connection between existing results, in this problem, yields a usable conclusion that the parts do not give separately. A targeted literature search is not proof that no precedent exists.
- *Testable increment.* A prediction or a cost saving that follows from the composition and can be checked on a real model against the nearest prior method at equal total cost.

<a id="s9-6"></a>

**Remark 9.6 (Contribution units and nearest prior work).**

**(a) Three contribution units.**

1. **Problem and representation.** Experience sufficiency is defined relative to a declared test class that stays closed under future learning operations, not relative to the current deployment alone. Responses at a checkpoint factor through the change of the next predictive state ([Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12)). This puts message compression ([Theorem 4.9](04-09_spectral_sufficiency.md#s4-9)) and learner memory ([Theorem 7.1](07-01_state_channel_decomposition.md#s7-1)) in one object. The constructions of [Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11) and [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6) show that matching current behaviour, or the immediate parameter change, can lose information that later training makes visible.
2. **Implementation and cost.** For an interface shared across checkpoints with state-conditional feedback differences, the framework gives reuse bounds ([Proposition 8.13](08-13_shared_features_across_checkpoints.md#s8-13)), an anchor transfer bound with an executable drift check ([Proposition 8.14](08-14_anchor_reuse_drift_certificate.md#s8-14)), the run count of a finite catalog ([Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12)) and an unbiased two-fidelity estimator with an exact cost condition ([Proposition 8.15](08-15_two_fidelity_response_estimation.md#s8-15)). Together they say when amortization pays and when it should be abandoned. Whether it pays on a real model is untested.
3. **Acquisition and compilation.** Uncertainty about the task consequences that the tests see, and the convex hull of the legal feedback, enter one radius, whose two lower bounds tell an information gap from a realizability gap ([Propositions 6.14](06-14_convex_realizability_legal_pool.md#s6-14) and [6.16](06-16_information_compilation_radius.md#s6-16)). Under per-test calibration, closure and coverage, the radius connects to the final training error ([Corollary 8.16](08-16_composed_guarantee.md#s8-16)). The linear-Gaussian design ([Proposition 6.15](06-15_linear_gaussian_target.md#s6-15)) is a computable special case, not the general rule.

None of the three is a claim of new mathematics, and none is a claim that the whole is already established as original. The identities of [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2) and [Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4) are bookkeeping tools and are not counted as contributions.

**(b) Nearest prior work.**

| Line of work | What must be conceded | What this framework must still show |
|---|---|---|
| Machine teaching ([Zhu, 2015](#ref-zhu2015machine); [Liu et al., 2017](#ref-liu2017iterative); [Lessard, Zhang & Zhu, 2019](#ref-lessard2019optimal)) | Teaching depends on the learner; examples are chosen iteratively for the learner's update rule; sequential teaching has already been posed as an optimal-control problem | The contribution cannot be "first to choose feedback by the learner" or "first to treat training as a controlled system". It must show that combining evidence about an unknown environment, sufficiency for continued training, legal compilation and cost-accounted maintenance changes decisions |
| Datamodels and TRAK ([Ilyas et al., 2022](#ref-ilyas2022datamodels); [Park et al., 2023](#ref-park2023trak)) | Counterfactual prediction of model outputs from changes of the training data; TRAK addresses computational feasibility at scale | A counterfactual response matrix is not a new class of object. The framework must separate a one-shot data attribution from a state-conditional, multi-round interface over legal feedback, and measure the cost |
| Harmonic analysis of datamodels ([Saunshi et al., 2023](#ref-saunshi2023understanding)) | Fourier coefficients and noise stability already analyse datamodels, including residual estimation without fitting a full datamodel | Not the first spectral view of data influence. The response-gain spectrum of [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9) differs from Boolean Fourier frequency, and the estimation methods must be compared concretely |
| Sketching a learning algorithm ([Gunn, 2026](#ref-gunn2026sketch)) | Compressible prediction of data-deletion counterfactuals from higher-order derivatives of the training algorithm, with a conditional computational guarantee | Compare task, access model and assumptions. Its stability condition on higher-order Taylor tensors is not assumed here; it already has a cost theory, which this framework does not yet match |
| Value-directed compression, predictive-state representations and their TD learning ([Poupart & Boutilier, 2002](#ref-poupart2002value); [Littman, Sutton & Singh, 2001](#ref-littman2001predictive); [Singh, James & Rudary, 2004](#ref-singh2004predictive); [Boots & Gordon, 2010](#ref-boots2010predictive)) | Tests as state, value-relevant compression and dynamic closure all have precedents | The difference lies in the complete learning process and the acquisition, compilation and reuse constraints. [Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12) is the lemma that unifies these constraints, not a rediscovery of predictive states |
| Value equivalence and task-sufficient world models ([Grimm et al., 2020](#ref-grimm2020value); [Feng et al., 2026](#ref-feng2026learning)) | Model sufficiency relative to a task or function class; active exploration for task-sufficient world models | Show that learning updates and optimizer memory change the target interface and the acquisition choices; renaming sufficiency "learner-relative" is not enough |
| LESS, GRACE, Adam data Shapley and metagradients ([Xia et al., 2024](#ref-xia2024less); [Panigrahi et al., 2025](#ref-panigrahi2025good); [Ding et al., 2026](#ref-ding2026inrun); [Engstrom et al., 2025](#ref-engstrom2025optimizing)) | The value of data or of a teacher can depend on the training process; Adam has been treated specifically | The memory construction of [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6) separates only interfaces that read the immediate parameter change. Any algorithmic advantage needs a correct adaptation of each method and an equal-total-cost comparison |
| Performance difference and the simulation lemma ([Kakade & Langford, 2002](#ref-kakade2002approximately)) | Decomposing differences of policies or models through value recursions and residuals | [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2) and [Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4) are basic tools, not core contributions |
| Doubly robust off-policy evaluation ([Jiang & Li, 2016](#ref-jiang2016doubly)) | A value model plus a correction term, with matching lower bounds in some cases | The value-model route to an estimable composed guarantee must be compared with it; any increment must come from structure of the learning system that lowers the cost of decisions over many candidates and checkpoints |
| Multifidelity and multilevel Monte Carlo ([Peherstorfer, Willcox & Gunzburger, 2016](#ref-peherstorfer2016optimal); [Giles, 2015](#ref-giles2015multilevel)) | Cheap approximations corrected by expensive residuals, with optimal allocation | The control-variate formula is not new. The candidate increment is its integration with response interfaces, the complete learner state and reuse across checkpoints |

**(c) What is missing.**

- No theorem shows an advantage, at equal total cost, over LESS, TRAK or datamodels; [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6) is a mechanism illustration, not such a separation.
- The strict versions of the reuse and composed guarantees are valid but, at the measured scale, vacuous ([Proposition 8.14](08-14_anchor_reuse_drift_certificate.md#s8-14), [Corollary 8.16](08-16_composed_guarantee.md#s8-16)). That is a property of the estimators and inequalities used, not a proof that certification is impossible. An estimation-cost theory must keep apart the gain of one fixed protocol pair, simultaneous certification of every local response, and sufficiency for a class of future protocols.
- The novelty of the composition is therefore to be judged by whether it yields testable increments: a gap diagnosis that changes a real decision, a measurable memory effect at a warm checkpoint (T1, [Remark 9.4](09-04_scope_of_evidence.md#s9-4)), or a reuse or two-fidelity saving on a real model. Until such results exist, the three units are proposals to be tested, not established contributions.

**What it says.** The framework's contribution is a composition of classical results into a problem about continued-training sufficiency, its cost and its legal implementation. Each nearest line of prior work already owns part of the ground, and the framework must still show testable increments over each.

**Role in the framework.**

- It fixes what the framework claims as new and what it concedes, so that later results can be read against the right baseline.
- It turns the novelty question into experiments: each unit names the decision it should change, and [Remark 9.4](09-04_scope_of_evidence.md#s9-4) records whether the tests so far support it.

**Uses.**

- [Propositions 4.11–4.12](04-11_static_not_dynamic_closure_error.md#s4-11), [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9), [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1), [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6): unit 1.
- [Propositions 8.12–8.15](08-12_finite_catalog_spectral_certificate.md#s8-12): unit 2.
- [Propositions 6.14–6.16](06-14_convex_realizability_legal_pool.md#s6-14), [Corollary 8.16](08-16_composed_guarantee.md#s8-16): unit 3.
- [Remark 9.5](09-05_assumption_audit.md#s9-5): the conditions each unit rests on.

**Used by.**

- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2): the method that embodies the three units.

## Related work

- The comparisons in the table above are the related work of this remark; each cited work is the primary source of the conceded part.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-boots2010predictive"></a>

**Byron Boots, Geoffrey J. Gordon (2010).** Predictive state temporal difference learning. *Advances in Neural Information Processing Systems 23 (NIPS)*, Curran Associates. [proceedings.neurips.cc/paper/2010/hash/3473decccb0509fb264818a7512a8b9b](https://proceedings.neurips.cc/paper_files/paper/2010/hash/3473decccb0509fb264818a7512a8b9b-Abstract.html).

<a id="ref-ding2026inrun"></a>

**Meng Ding, Zeqing Zhang, Di Wang, Lijie Hu (2026).** In-Run Data Shapley for Adam optimizer. arXiv preprint. arXiv [2602.00329](https://arxiv.org/abs/2602.00329).

<a id="ref-engstrom2025optimizing"></a>

**Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, Aleksander Madry (2025).** Optimizing ML training with metagradient descent. arXiv preprint. arXiv [2503.13751](https://arxiv.org/abs/2503.13751).

<a id="ref-feng2026learning"></a>

**Fan Feng, Yujia Zheng, Minghao Fu, Yongqiang Chen, Guangyi Chen, Kevin Patrick Murphy, Biwei Huang, Kun Zhang (2026).** Learning task-sufficient world models by synergizing agentic exploration and structured modeling. *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, PMLR 306:30616–30641. [proceedings.mlr.press/v306/feng26aa](https://proceedings.mlr.press/v306/feng26aa.html); arXiv [2607.04409](https://arxiv.org/abs/2607.04409).

<a id="ref-giles2015multilevel"></a>

**Michael B. Giles (2015).** Multilevel Monte Carlo methods. *Acta Numerica* 24, 259–328. DOI [10.1017/S096249291500001X](https://doi.org/10.1017/S096249291500001X).

<a id="ref-grimm2020value"></a>

**Christopher Grimm, André Barreto, Satinder Singh, David Silver (2020).** The value equivalence principle for model-based reinforcement learning. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 5541–5552. arXiv [2011.03506](https://arxiv.org/abs/2011.03506).

<a id="ref-gunn2026sketch"></a>

**Sam Gunn (2026).** How to sketch a learning algorithm. *arXiv preprint*. arXiv [2604.07328](https://arxiv.org/abs/2604.07328).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").

<a id="ref-jiang2016doubly"></a>

**Nan Jiang, Lihong Li (2016).** Doubly robust off-policy value evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48:652–661. [proceedings.mlr.press/v48/jiang16](https://proceedings.mlr.press/v48/jiang16.html).

<a id="ref-kakade2002approximately"></a>

**Sham Kakade, John Langford (2002).** Approximately optimal approximate reinforcement learning. *Proceedings of the 19th International Conference on Machine Learning (ICML)*, 267–274, Morgan Kaufmann. [dl.acm.org/doi/10.5555/645531.656005](https://dl.acm.org/doi/10.5555/645531.656005).

<a id="ref-lessard2019optimal"></a>

**Laurent Lessard, Xuezhou Zhang, Xiaojin Zhu (2019).** An optimal control approach to sequential machine teaching. *Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 89:2495–2503. [proceedings.mlr.press/v89/lessard19a](https://proceedings.mlr.press/v89/lessard19a.html).

<a id="ref-littman2001predictive"></a>

**Michael L. Littman, Richard S. Sutton, Satinder Singh (2001).** Predictive representations of state. *Advances in Neural Information Processing Systems 14 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html).

<a id="ref-liu2017iterative"></a>

**Weiyang Liu, Bo Dai, Ahmad Humayun, Charlene Tay, Chen Yu, Linda B. Smith, James M. Rehg, Le Song (2017).** Iterative machine teaching. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:2149–2158. [proceedings.mlr.press/v70/liu17b](https://proceedings.mlr.press/v70/liu17b.html).

<a id="ref-panigrahi2025good"></a>

**Abhishek Panigrahi, Bingbin Liu, Sadhika Malladi, Sham Kakade, Surbhi Goel (2025).** In good GRACEs: Principled teacher selection for knowledge distillation. arXiv preprint. arXiv [2511.02833](https://arxiv.org/abs/2511.02833).

<a id="ref-park2023trak"></a>

**Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, Aleksander Madry (2023).** TRAK: Attributing model behavior at scale. *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202:27074–27113. [proceedings.mlr.press/v202/park23c](https://proceedings.mlr.press/v202/park23c.html).

<a id="ref-peherstorfer2016optimal"></a>

**Benjamin Peherstorfer, Karen Willcox, Max Gunzburger (2016).** Optimal model management for multifidelity Monte Carlo estimation. *SIAM Journal on Scientific Computing* 38, no. 5 (section A, Methods and Algorithms for Scientific Computing). DOI [10.1137/15M1046472](https://doi.org/10.1137/15M1046472).

<a id="ref-poupart2002value"></a>

**Pascal Poupart, Craig Boutilier (2002).** Value-directed compression of POMDPs. *Advances in Neural Information Processing Systems 15 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d](https://proceedings.neurips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html).

<a id="ref-saunshi2023understanding"></a>

**Nikunj Saunshi, Arushi Gupta, Mark Braverman, Sanjeev Arora (2023).** Understanding influence functions and datamodels via harmonic analysis. *11th International Conference on Learning Representations (ICLR)*. [openreview.net/forum?id=cxCEOSF99f](https://openreview.net/forum?id=cxCEOSF99f); arXiv [2210.01072](https://arxiv.org/abs/2210.01072).

<a id="ref-singh2004predictive"></a>

**Satinder Singh, Michael R. James, Matthew R. Rudary (2004).** Predictive state representations: A new theory for modeling dynamical systems. *Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI)*, 512–519, AUAI Press. arXiv [1207.4167](https://arxiv.org/abs/1207.4167).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).

<a id="ref-zhu2015machine"></a>

**Xiaojin Zhu (2015).** Machine teaching: An inverse problem to machine learning and an approach toward optimal education. *Proceedings of the 29th AAAI Conference on Artificial Intelligence (AAAI)* 29, no. 1. DOI [10.1609/aaai.v29i1.9761](https://doi.org/10.1609/aaai.v29i1.9761).
