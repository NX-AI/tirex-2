# TiRex-2 Intended Use, Limitations and EU AI Act Notice

- **Notice version:** 1.0
- **Date:** 22 September 2026
- **Applies to:** the public TiRex-2 repository, public model checkpoint `NX-AI/TiRex-2`, and associated public inference software
- **Licence:** Apache License 2.0

## 1. Purpose and status of this notice

This notice describes NXAI's intended purpose for the public TiRex-2 release, material model and deployment limitations, and expectations for downstream integration. It is product and regulatory information, not legal advice.

This notice does not amend, replace, or restrict the Apache License 2.0. Licence permissions and conditions are governed exclusively by the applicable licence text. Each actor remains responsible for determining and fulfilling the legal obligations applicable to its own role and use case.

## 2. Intended purpose

TiRex-2 is a time-series foundation model intended for zero-shot forecasting. The public release supports univariate and multivariate forecasting and can use past and future-known covariates. It is intended for research, evaluation, and non-high-risk decision support in applications such as demand, operations, logistics, retail, and predictive-maintenance analysis.

The public release produces point and probabilistic forecasts. Its standard probabilistic output comprises nine quantiles from 10% to 90%.

TiRex-2 is not intended to make autonomous decisions, operate machinery, or replace qualified human judgement. In predictive maintenance, its intended role is to support analysis and planning. It is not intended to act as a safety component, trigger an automatic shutdown, or provide the sole basis for a maintenance or safety decision.

## 3. Excluded intended uses

NXAI does not intend this release to be used for any prohibited AI practice under Article 5 of the EU AI Act.

NXAI clearly specifies that this release is not to be changed into, integrated into, or materially relied upon as a high-risk AI system within the meaning of Article 6 and Annexes I and III of the EU AI Act. This includes use as a safety component of a product covered by Annex I and use as the sole or determinative basis for decisions in high-risk areas such as critical infrastructure, education, employment, essential services, law enforcement, migration, or the administration of justice.

Whether a particular downstream application is high-risk depends on its intended purpose, functionality, and deployment context. Use in a sector mentioned in the AI Act is not, by itself, a complete classification.

This section defines NXAI's intended purpose. It is not a licence restriction and does not purport to transfer, exclude, or reallocate obligations imposed by law.

## 4. Model limitations

- Forecast quality depends on the relevance, quality, frequency, length, and preprocessing of the input series and covariates.
- Performance reported on research benchmarks does not establish performance, safety, or regulatory compliance for a specific operational deployment.
- Distribution shift, rare events, regime changes, missing or erroneous observations, and unsuitable covariates can materially reduce forecast quality.
- Quantile outputs express model-estimated forecast uncertainty. They are not a guarantee of real-world coverage or application-level calibration. Calibration must be evaluated on representative deployment data.
- TiRex-2 does not provide causal explanations or a built-in explanation of individual forecasts. External techniques may be evaluated by the integrator, but their suitability and correctness are not guaranteed by NXAI.
- The model does not provide application-level human oversight, access control, audit logging, fail-safe behaviour, or machinery control logic. These are integration responsibilities.
- The public release does not provide stateful incremental streaming inference. Each forecast recomputes the supplied context. Streaming and other PRO capabilities have separate functionality and documentation and require a separate deployment assessment.

## 5. Deployment and security limitations

The public HTTP API does not implement authentication. It must not be exposed directly to an untrusted network. Integrators should place it behind appropriate authentication, authorisation, transport encryption, rate limiting, network controls, and monitoring.

NXAI does not operate the public TiRex-2 release as a hosted model service and therefore does not receive or retain deployment inputs, outputs, or inference logs. The deployer is responsible for appropriate logging, retention, access control, security, data protection, and incident handling. Customer-specific development or training engagements are governed by their applicable contracts.

## 6. Integration and validation responsibilities

Before operational use, the integrating organisation should:

- define the intended purpose, users, operating environment, and foreseeable misuse of the resulting AI system;
- determine the organisation's role and the regulatory classification of the resulting AI system;
- validate accuracy, robustness, uncertainty calibration, latency, resource requirements, and failure behaviour on representative data;
- establish human review, override, escalation, and safe fallback procedures proportionate to the consequences of an incorrect forecast;
- prevent forecast outputs from directly controlling safety-relevant machinery without independently validated safeguards;
- identify and retain the model, software, configuration, and data-processing versions needed for traceability;
- implement suitable input/output monitoring and logging without recording personal, confidential, or security-sensitive data unnecessarily;
- review data rights, data protection, cybersecurity, and sector-specific requirements; and
- monitor performance and reassess the deployment when data, operating conditions, model versions, or intended purpose change.

## 7. Responsibilities under the EU AI Act

This notice does not constitute regulatory approval, a conformity assessment, or a binding classification of TiRex-2 or any downstream AI system.

Under Article 25 of the EU AI Act, a downstream actor may assume provider obligations when, for example, it places a high-risk AI system on the market under its own name, substantially modifies such a system, or changes its intended purpose so that it becomes high-risk. The application of Article 25 and any open-source-related provisions must be assessed for the specific release, actor, and deployment.

Each actor remains responsible for obligations applicable to its own role. NXAI's stated intended purpose does not remove obligations that the law assigns to NXAI, an integrator, a deployer, or another actor.

## 8. Further information

- Repository: <https://github.com/NX-AI/tirex-2>
- Documentation: <https://nx-ai.github.io/tirex-2/>
- Deployment guidance: <https://nx-ai.github.io/tirex-2/deployment/>
- Public model: <https://huggingface.co/NX-AI/TiRex-2>
- Apache License 2.0: <https://github.com/NX-AI/tirex-2/blob/main/LICENSE>
- EU AI Act, consolidated text: <https://eur-lex.europa.eu/eli/reg/2024/1689/2026-07-27/eng>
- Commission guidelines on general-purpose AI models: <https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-3>

- **Provider contact:** NXAI GmbH, Peter-Behrens-Platz 2, 4020 Linz, Austria
- **Company register:** FN 616894 y, Landesgericht Linz
- **VAT ID:** ATU80117419
- **Website:** <https://www.nx-ai.com>
- **Contact:** <contact@nx-ai.com>

## 9. Acknowledgement

This content has been advised by and developed in collaboration with the EUSAiR project team as part of a simulation of AI regulatory sandboxes, in which NXAI participated. EUSAiR is a CSA project funded by the European Commission, designed to assist the AI Office and National Competent Authorities in the planning and establishment of AI regulatory sandboxes. For more information, please visit the website: [eusair-project.eu](https://eusair-project.eu/)
