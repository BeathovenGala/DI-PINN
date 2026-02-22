# DI-PINN: Differentiable Inverse Physics-Informed Neural Network
### Reconstructing dark matter mass maps from real strong lensing images

---

## The Core Idea and Motivation

The study of dark matter through strong gravitational lensing has reached a pivotal juncture. While previous machine learning efforts (like the 2024 GSoC project LensPINN) successfully demonstrated the classification of dark matter substructures on simulated datasets, transitioning to real observational data remains a massive hurdle. 

Traditional analytic modeling tools (like Lenstronomy) provide high physical accuracy but suffer from computationally expensive Markov Chain Monte Carlo (MCMC) sampling, often requiring hours per lens. On the flip side, standard deep learning models offer incredible speed but lack the physical consistency required for true scientific validity when facing complex noise, point spread functions (PSF), and light contamination present in real-world surveys like HSC, HST, and the upcoming Vera C. Rubin Observatory (LSST).

To bridge this gap, I am developing a framework capable of rapid inference and discovery in real observational datasets. The goal is to move beyond simple classification and enable the precise localization of substructures and the quantification of their mass power spectrum.

Rather than just learning a class label, DI-PINN learns to predict continuous lensing parameters ($\theta_E, q, \phi, \gamma$) alongside the full convergence field, κ(x,y) (the projected mass density). With this map, I can:
- Locate subhalos by finding peaks in the κ field.
- Measure precise masses by integrating around those peaks.
- Understand dark matter morphology (e.g., Cold Dark Matter vs. Axion models).
- Extract classification, regression, and anomaly detection entirely as post-processing steps.

## Expanding Beyond Previous Methods

This framework represents a paradigm shift from "black-box prediction" to "physics-driven inverse modeling."

| Core Problem | Previous Work (LensPINN 2024) | DI-PINN (This Architecture) |
| --- | --- | --- |
| **Physics Engine** | Static Analytical Formula (limited to SIS profiles). | Differentiable Simulator (`caustics`) handling complex generic profiles. |
| **Input Data** | Trained and tested exclusively on Simulated Model II data. | Explicitly designed for real HSC/HST/LSST observations + Sim. |
| **Foreground Light** | Assumed negligible or already removed. | Active lens light removal via Multi-Gaussian Expansion (MGE). |
| **Scientific Value** | Outputs a categorical class label (e.g., "Axion" vs "CDM"). | Mass parameters, reconstructed source image, and the power spectrum. |
| **Confidence Level** | Deterministic (Softmax). | Probabilistic (Bayesian Deep Ensembles) providing uncertainty maps. |

This evolution moves the project from simply "labelling the jar" to actually "counting the marbles inside."

## The Key Components

Every piece of this framework integrates directly into the neural network's computation graph to handle real data and enforce physics.

| Component | Purpose | Why it's critical here |
| --- | --- | --- |
| **`caustics`** | Differentiable Ray-Tracing | Provides a fully differentiable simulator where gradients flow from the reconstructed image back through the physical lensing equations to update network parameters. |
| **Multi-Gaussian Expansion (MGE)** | Automated Light Subtraction | Real galaxy light drowns the faint lensed arcs. MGE models and isolates them mathematically. |
| **PI-AdaMatch** | Domain Adaptation | Enforces a *Self-Consistent Lensing Loss* on pseudo-labels for real images, ensuring predictions satisfy gravity during Domain Adaptation. |
| **Bayesian Deep Ensembles** | Uncertainty Quantification | Calculates the pixel-wise variance of source reconstructions. High-variance regions serve as candidate locations for dark matter subhalos. |

## Architecture Data Flow

The architecture is explicitly designed so the signal flows from raw input down to a physically constrained mass map.

```mermaid
flowchart TB
    subgraph ENCODING["Encoding Pipeline"]
        direction LR
        I["Input Image\nI_obs(θ)\n(real/simulated)"] --> PRE["Preprocessing\n- MGE Light Subtraction\n- Differentiable PSF"]
        PRE --> ENC["LensPINN_large ViT Encoder\nPredicts Lensing\nPotential Ψ(x,y)"]
    end
    
    ENC -->|Ψ(x,y)| LENS["Differentiable Lensing Layer (caustics)\n- α = ∇Ψ (deflection angles)\n- β = θ - α (source coordinates)\n- I_rec = ray-trace(S(β))"]
    
    LENS --> LOSS["Loss Computation\nℒ = ℒ_data + λ₁ℒ_poisson + λ₂ℒ_reg"]
    
    LOSS --> OUT["Outputs\n- Precise κ map\n- Reconstructed Source\n- Uncertainty Maps"]
```

## The Underlying Physics

There are a few key equations that the network is forced to obey. By baking these in via the `caustics` library, the network is prevented from hallucinating physically impossible mass distributions.

| Equation | Mathematical Form | What it Enforces |
| --- | --- | --- |
| **Poisson** | ∇²Ψ = 2κ | The fundamental relationship tying mass to gravitational potential. |
| **Deflection** | α = ∇Ψ | The potential gradient dictates how the light actually bends. |
| **Lens Equation** | β = θ - α(θ) | The geometric ray-tracing mapping from the image plane back to the source plane. |
| **Data Fidelity** | ℒ_{data} = ‖I_{obs} - I_{rec}‖² | The image reconstructed through the forward pass must match what the telescope observed. |
| **Physics Consistency** | ℒ_{poisson} = ‖∇²Ψ - 2κ‖² | The network's internal representation strictly satisfies the laws of gravity. |

## Anticipated Challenges and Mitigation Strategies

Applying inverse physics models to real space data introduces several roadblocks. Here is how I constructed the pipeline to mitigate them:

| Roadblock | Mitigation Strategy |
| --- | --- |
| **MGE itself isn't differentiable.** | It is run as an automated one-off pre-processing step using `mge_fit`. Since lens light isn't part of the dark matter inference, treating it outside the gradient graph works perfectly. |
| **Predicting a full 2D potential field can be unstable.** | The U-Net style decoder includes skip connections to naturally encourage smoothness, alongside a Laplacian regularization term to penalize unphysical wave oscillations. |
| **Memory bottlenecks during ray-tracing over full grids.** | Handled via mixed-precision training (float16) and gradient checkpointing. The `caustics` backend is highly GPU-optimized to keep memory overhead manageable. |
| **PI-AdaMatch producing low-quality pseudo-labels early on.** | Training begins with a pure simulation warm-up. Real data is gradually introduced, and pseudo-labels are strictly filtered out if their reconstruction error is above a designated threshold. |
| **Securing accurate uncertainty calibration.** | A deep ensemble of independently initialized models provides epistemic uncertainty. The pixel-wise variance of these models explicitly highlights regions where the prediction is unreliable. |
| **Lack of a ground truth κ map for real observations.** | Model validation is performed by comparing predictions against Lenstronomy MCMC fits on "Gold Standard" lenses. Injection-recovery tests confirm the reliable detection of synthetic subhalos. |

## Current Progress and Proof of Concept

This framework is not just a theoretical proposal—the core minimum viable code (MVC) has already been built, tested, and validated. 

| Component | Status | Details |
| --- | --- | --- |
| **Synthetic Data Pipeline** | Complete | Generates complex SIS halos along with random subhalos on the fly. |
| **U-Net Baseline Model** | Complete | A functioning encoder-decoder architecture handling complex spatial features. |
| **Physics Modules** | Complete | The integrated Poisson solver and ray-tracing logic have been built and unit-tested to ensure exact physics calculations. |
| **Training Pipeline** | Complete | Implements supervised looping with MSE loss, live checkpointing, and dynamic plotting. |
| **Trained Weights** | Complete | Holds the baseline trained model (`mvc_unet.pth`) capable of successful κ reconstruction on simplified data. |

This foundational code proved that the data pipeline, loss structure, and theoretical integration fundamentally work. Development now shifts to scaling this solid base by wrapping it fully around the `caustics` engine and introducing the noise, foreground glare, and underlying uncertainty of actual telescope observations.
