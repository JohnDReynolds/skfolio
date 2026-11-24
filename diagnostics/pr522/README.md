# PR522 paired numerical investigation

This diagnostic branch compares immutable accepted and PR source commits, plus one explicitly experimental evaluator. It does not revise the published PR branch.

Each profile uses one native runner/environment and the exact recorded package versions from the failed PR job. The identical PR entropy-pooling test module, including its unchanged assertions, is used for all variants. Existing and newly added tests are distinguishable by name. Original failing results are evidence, not passing gates. A green diagnostic job means the harness completed; consult each variant's pytest result, JUnit, and logs for actual failures.

The experimental evaluator retains the original unnormalized objective, multiplier count, bounds and TNC options. It rewrites exp(u) as 1+expm1(u) and cancels the normalization terms algebraically. It is an assessment prototype, not an accepted fix. No test tolerance, solver default or public API is changed. Numerical failure records and solver traces are retained for comparison.

Two additional explicitly experimental controls retain each evaluator and change only default TNC stopping options to ftol=0, xtol=1e-12, gtol=1e-10; caller-supplied options are retained unchanged. These are design controls, not approved production defaults. The expm1-only prototype already fails three local cases; it is preserved as a rejected candidate, not described as passing. No further parameter search is performed.

## Approved normalized-dual prototype

The follow-up run compares accepted, PR, and normalized evaluators using the same124 original test cases. The old normalization-multiplier-only callback expectations are kept in that run and will fail under the explicitly changed coordinates; their results are not hidden or relabeled as passing. An additional normalized-contract run independently checks the new objective/gradient, old full-length scale/offset adapter and complete public posterior/control. All existing financial/projection tolerances stay unchanged. The prototype retains TNC defaults and fixed-moment bounds. The normalization slot of provided arrays is intentionally ignored, while caller objects stay unchanged. This compatibility policy is approved for assessment only, not yet production publication.
