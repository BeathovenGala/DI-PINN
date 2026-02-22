# DI-PINN: Differentiable Inverse Physics-Informed Neural Network
### Reconstructing dark matter mass maps from real strong lensing images

---

## Introduction

Recent studies, such as the 2020 work *Deep Learning the Morphology of Dark Matter Substructure*, established that convolutional neural networks could reliably distinguish between different types of dark matter—such as CDM subhalos versus axion vortices—in simulated strong lensing images. Treating this as a classification problem was identified as an "intermediate step" before the ultimate goal: determining the position, mass, and other physical properties of individual substructures.

The 2021 follow-up, *Decoding Dark Matter Substructure without Supervision*, further demonstrated that unsupervised models, particularly Adversarial Autoencoders, could serve as highly effective anomaly detectors. Crucially, the reconstruction MSE loss was shown to inherently encode information about the location of substructures, pointing toward the possibility of "using this data to invert the lens equation to produce the distribution of substructure mass on the lensing plane."

Most recently, the 2024 *LensPINN* architecture demonstrated the efficacy of explicitly integrating the gravitational lensing equation into a ViT-CNN framework, achieving exceptional convergence and accuracy by enforcing physics-informed preprocessing and lensing inversion.

Analyzing these developments sequentially reveals a clear methodological trajectory. Successive models move progressively toward a common destination: theory-agnostic, spatial mass maps of dark matter substructures in real telescope images. While the physical models are well-understood, the ML architectures are proven on simulations, and observational datasets are imminent with the Vera C. Rubin Observatory (LSST) and Euclid, the framework to synthesize these breakthroughs for application on complex, real-world observational data remains to be established.

This proposal introduces DI-PINN (Differentiable Inverse Physics-Informed Neural Network). DI-PINN is formulated to explicitly invert the lens equation to produce the distribution of substructure mass on the lensing plane. Building directly upon the foundational physics-informed approaches of its predecessors, DI-PINN extends the architecture in three critical directions necessary for observational application:

- **Continuous 2D Substructure Mapping:** Predicting the full 2D lensing potential Ψ(x,y) rather than relying on scalar parameters or discrete classification schemes.
- **Observational Data Readiness:** Handling real observational complexities through Multi-Gaussian Expansion (MGE) lens light subtraction and PI-AdaMatch domain adaptation.
- **Uncertainty Quantification:** Providing calibrated Bayesian uncertainty mapping to establish statistically rigorous confidence intervals for the network's predictions.

The core architectural paradigm mirrors the progression from black-box to physics-informed modeling: instead of learning a direct mapping from images to target labels, a differentiable physics engine (`caustics`) is embedded as a fixed, deterministic layer within the network. The trainable components learn to predict the lensing potential Ψ(x,y); the physics layer then performs exact ray-tracing, strictly enforces the Poisson equation ∇²Ψ = 2κ, and reconstructs the observed image. Consequently, gradients flow through this entire physical pipeline, ensuring that every weight update iteratively drives the predictions toward solutions that satisfy Einstein's field equations.

## Architecture Data Flow

The architecture is explicitly designed so the signal flows from raw input down to a physically constrained mass map.

```mermaid
flowchart TB
    subgraph ENCODING["Encoding Pipeline"]
        direction LR
        I["Input Image\nI_obs(θ)\n(real/simulated)"] --> PRE["Preprocessing\n- MGE Light Subtraction\n- Differentiable PSF"]
        PRE --> ENC["LensPINN_large ViT Encoder\nPredicts Lensing\nPotential Ψ(x,y)"]
    end
    
    ENC -->|Ψ_x,y| LENS["Differentiable Lensing Layer (caustics)<br>α = ∇Ψ (deflection angles)<br>β = θ - α (source coordinates)<br>I_rec = ray-trace(S(β))"]
    
    LENS --> LOSS["Loss Computation<br>ℒ = ℒ_data + λ₁ℒ_poisson + λ₂ℒ_reg"]
    
    LOSS --> OUT["Outputs<br>- Precise κ map<br>- Reconstructed Source<br>- Uncertainty Maps"]
```

## Key Components

This framework represents a paradigm shift from "black-box prediction" to "physics-driven inverse modeling." Every piece of this framework integrates directly into the neural network's computation graph to handle real data and enforce physics.

| Component / Feature | Previous Work (LensPINN 2024) | DI-PINN (This Architecture) | Why it's critical here |
| --- | --- | --- | --- |
| **Physics Engine** | Static Analytical Formula (limited to SIS profiles). | Differentiable Simulator (`caustics`) handling complex generic profiles. | Provides a fully differentiable simulator where gradients flow from the reconstructed image back through the physical lensing equations to update network parameters. |
| **Input Data** | Trained and tested exclusively on Simulated Model II data. | Explicitly designed for real HSC/HST/LSST observations + Sim. | Enables the model to bridge the gap between idealized models and real telescope findings. |
| **Foreground Light** | Assumed negligible or already removed. | Active lens light removal via Multi-Gaussian Expansion (MGE). | Real galaxy light drowns the faint lensed arcs. MGE models and isolates them mathematically. |
| **Scientific Value** | Outputs a categorical class label (e.g., "Axion" vs "CDM"). | Mass parameters, reconstructed source image, and the power spectrum. | Moves beyond categories to actual scalable physical parameters. |
| **Confidence Level** | Deterministic (Softmax). | Probabilistic (Bayesian Deep Ensembles) providing uncertainty maps. | Calculates the pixel-wise variance of source reconstructions. High-variance regions serve as candidate locations for dark matter subhalos. |
| **Domain Adaptation** | None. | PI-AdaMatch. | Enforces a *Self-Consistent Lensing Loss* on pseudo-labels for real images, ensuring predictions satisfy gravity during Domain Adaptation. |

*This evolution moves the project from simply "labelling the jar" to actually "counting the marbles inside."*

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

## Current MVC Outline

THe current flow for the MVC is as follows:

*   **Synthetic Data Pipeline:** Generates complex SIS halos along with random subhalos on the fly.
*   **U-Net Baseline Model:** A functioning encoder-decoder architecture handling complex spatial features.
*   **Physics Modules:** The integrated Poisson solver and ray-tracing logic have been built and unit-tested to ensure exact physics calculations.
*   **Training Pipeline:** Implements supervised looping with MSE loss, live checkpointing, and dynamic plotting.
*   **Trained Weights:** Holds the baseline trained model (`mvc_unet.pth`) capable of successful κ reconstruction on simplified data.

Development now shifts to scaling this solid base by wrapping it fully around the `caustics` engine and introducing the noise, foreground glare, and underlying uncertainty of actual telescope observations.
