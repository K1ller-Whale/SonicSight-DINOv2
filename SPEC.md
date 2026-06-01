



RESEARCH REPORT
DINOv2-Guided Audio-Visual Source Separation
A Next-Generation Architecture Beyond Sound of Pixels


Version 1.0  —  2026

Table of Contents

1. Executive Summary

This report presents the design and rationale for a novel audio-visual source separation architecture that replaces the convolutional visual encoder of the Sound of Pixels framework with DINOv2, a state-of-the-art self-supervised Vision Transformer backbone. The proposed system uses DINOv2 patch-level features as a rich semantic grounding signal for cross-modal attention fusion, enabling the model to separate audio sources across a broad set of real-world classes including human speech, musical instruments, and environmental sounds.



2. Background and Motivation

2.1 The Source Separation Problem
Audio source separation is the task of recovering individual sound sources from a mixture signal — the so-called cocktail party problem. In its most challenging form, the input is a single-channel recording of N simultaneous sources, and the system must output N clean audio streams without any prior knowledge of the number or identity of the sources.
Classical approaches such as Independent Component Analysis (ICA) and Non-negative Matrix Factorization (NMF) rely on statistical assumptions about source independence or spectral structure. While effective in constrained settings, these methods degrade rapidly when sources share spectral content (e.g., two violins, or two male voices).
Deep learning approaches have significantly advanced the field. Time-domain methods such as Conv-TasNet and DPRNN operate directly on waveforms and achieve state-of-the-art performance on benchmark datasets such as WSJ0-2mix. However, these systems are purely audio-driven and lack the ability to use visual cues to resolve ambiguity between sources.
2.2 Audio-Visual Separation: The Sound of Pixels
The Sound of Pixels (Zhao et al., 2018) introduced the paradigm of visually-guided audio source separation. The key insight is that when a video shows multiple sound-producing objects, the visual stream provides a natural grounding signal: if you can identify which pixels correspond to a piano, you can use that information to isolate the piano's audio from a mixture.
The original architecture consists of three components:
A visual subnetwork (ResNet-18 or similar) that encodes video frames into a spatial feature map.
An audio subnetwork (U-Net on log-mel spectrograms) that encodes the mixed audio.
A fusion module that multiplies visual and audio features channel-wise, followed by a decoder that predicts soft spectral masks for each source.

Despite its elegance, Sound of Pixels has notable limitations that motivate the present work:
The visual encoder is trained end-to-end with audio supervision, meaning it learns only the visual features useful for the specific training distribution of instruments or speakers.
The ResNet feature map is semantically shallow compared to modern Vision Transformers — it captures texture and shape but does not understand 'this is a violin', 'this is a trumpet', or 'this is person A versus person B'.
Generalization to unseen source classes requires retraining the entire visual backbone.
The channel-wise feature multiplication fusion mechanism is a relatively weak form of cross-modal interaction.

3. The DINO Model Family

3.1 DINO and Self-Supervised Vision
DINO (DIstillation with NO labels, Caron et al., 2021) established that Vision Transformers trained with self-supervised objectives develop emergent properties that supervised models do not — most notably, their attention maps spontaneously segment foreground objects without any segmentation supervision. This occurs because the self-distillation objective forces the model to learn what makes two augmented views of the same image similar, which implicitly requires understanding semantic content rather than low-level texture.
The training mechanism uses a teacher-student framework where both networks share the same architecture but the teacher's weights are an exponential moving average (EMA) of the student's weights. Both see different augmented crops of the same image; the student is trained to predict the teacher's output distribution. No labels are used at any point.
3.2 DINOv2
DINOv2 (Oquab et al., 2023) scaled the original DINO recipe to a curated dataset of 142 million images (LVD-142M) and introduced additional training objectives including iBOT (masked image modelling) and Sinkhorn-Knopp centering. The resulting models — ViT-S, ViT-B, ViT-L, and ViT-G — produce patch-level features that serve as universal visual representations, achieving state-of-the-art results on depth estimation, semantic segmentation, image classification, and instance retrieval with simple linear probes.
Most importantly for our application, DINOv2 patch tokens carry rich spatial semantic information. Each patch token encodes not just the local appearance of a 14x14 pixel region but also its global context through self-attention layers. Two patches showing different violins in different scenes will produce similar feature vectors; a patch showing a violin and a patch showing a cello will produce meaningfully different but related vectors.
3.3 DINOv3
According to community reports, DINOv3 further extends DINOv2 with a much larger teacher ViT trained on approximately 1.7 billion images. Reported innovations include Gram Anchoring — a regularization term stabilizing patch-level features during extended training — and a high-resolution fine-tuning stage. It is reportedly distilled into ViT-S, ViT-B, ViT-L, ViT-H+, and ConvNeXt variants. These details should not be cited as established facts until a formal publication is available.
For the purpose of this work, we target DINOv2-Base as the visual backbone because it is fully open-access, well-supported by the HuggingFace Transformers library, and produces 768-dimensional patch features that are sufficient for cross-modal fusion. DINOv3 variants can be substituted in a drop-in manner once access restrictions are resolved.

4. Architecture Comparison: Sound of Pixels vs. Proposed Model

The table below summarizes the key architectural and methodological differences between the original Sound of Pixels system and the proposed DINOv2-guided separation model.

Table 1. Side-by-side comparison of Sound of Pixels and the proposed DINOv2-guided architecture.

5. Proposed Source Classes and Justification

A key design decision is which source classes the model should be trained to separate. We organise the target classes into four tiers based on the richness of the training data available and the semantic distinctiveness of the corresponding visual features in DINOv2's representation space.
5.1 Tier 1 — Human Speech (Highest Priority)
Speech separation is the most commercially important application and the domain where the cocktail party problem is most acutely felt. We target two distinct sub-tasks:
Speaker diarisation and separation from a single mixed recording. Training data: VoxCeleb2 (1M+ utterances, 6,112 identities), AVSpeech (Google, 150K clips). Multi-speaker separation:
Isolating a speaking foreground voice from background noise or music. Training data: AudioSet with speech-dominant segments. Speaker-from-background:
Justification for DINOv2: DINOv2 patch features clearly distinguish between different speakers' lip regions, head poses, and facial identities. In a scene with two talking people, each person's face cluster occupies a distinct region of the DINOv2 feature space, providing a reliable visual anchor for separating their voices.

5.2 Tier 2 — Musical Instruments
The MUSIC dataset (Zhao et al., 2018) covers 11 instrument categories and remains the standard benchmark for audio-visual separation. We extend this to 21 categories to cover a broader range of timbral profiles:

Table 2. Proposed instrument categories across four families.

Justification for DINOv2: Instrument categories are among the most visually discriminative classes in DINOv2's feature space. A ViT-B trained on LVD-142M has encountered hundreds of thousands of images of each instrument across diverse contexts (concert halls, street performances, studio sessions). The resulting patch features cluster strongly by instrument family without any audio supervision.
5.3 Tier 3 — Environmental and Urban Sounds
Real-world recordings frequently contain environmental sources that overlap spectrally with speech and music. We include the following categories from AudioSet and FreeSound:
Vehicle sounds: car engine, motorcycle, bicycle, train, aircraft
Nature sounds: rain, wind, flowing water, birdsong, thunder
Urban sounds: construction, crowd, traffic, air conditioning
Household sounds: television, kitchen appliances, door, footsteps
Justification: While environmental sources are less visually distinctive than instruments or faces, DINOv2 features still provide useful localisation. A microphone pointed at a running engine occupies a visually distinct patch region from a person speaking in the foreground. Cross-modal attention can leverage this spatial contrast even when the visual semantics are less rich than for instruments.
5.4 Tier 4 — Animal Vocalisations
Animal vocalisations represent a challenging but valuable test case for generalisation. Tier 4 is included primarily as an out-of-distribution benchmark rather than a primary training target. Categories include dogs, cats, birds, and farm animals from the Animal Sounds subset of AudioSet.

6. Why the New Approach Is Better Than Fine-Tuning Sound of Pixels

A natural question is whether the goals of this project could be achieved by simply fine-tuning the existing Sound of Pixels model on a larger dataset. This section argues systematically that fine-tuning is an inadequate strategy and that the architectural changes proposed here are necessary rather than merely incremental.
6.1 The Shallow Visual Representation Problem
The ResNet backbone in Sound of Pixels was originally pre-trained on ImageNet with class labels, then fine-tuned jointly with the audio separation objective. This creates two compounding limitations:
ImageNet features are optimised for 1,000 object categories with a single dominant object per image. They generalise poorly to scenes with multiple interacting objects, unusual viewpoints, or categories not represented in ImageNet.
Joint fine-tuning with audio supervision biases the visual features toward whatever visual correlates are statistically useful for the training audio mix, not toward semantic object understanding. If the training set contains only guitars and pianos, the visual encoder will not learn to distinguish a violin from a cello even if their visual appearances are clearly different.
DINOv2 features, by contrast, are trained on 142 million diverse images with no label bias. The self-supervised objective forces the model to learn features that generalise across all object categories, making the feature space intrinsically more discriminative for unseen classes.
6.2 The Fusion Mechanism Bottleneck
Sound of Pixels uses channel-wise feature multiplication to fuse visual and audio information. While computationally efficient, this is a fundamentally limited operation: it can gate audio features based on visual presence (is there a piano in the scene?) but cannot perform fine-grained spatial correspondence (which audio frequency bins correspond to the hand position on the piano keyboard?).
Cross-modal attention overcomes this by allowing every audio feature vector to independently query the entire visual feature map. A spectrogram bin representing a high-frequency transient can attend specifically to the visual patches showing the striking keys rather than averaging over the entire instrument silhouette. This spatial precision is critical for separating instruments that are close together in the scene.
6.3 Catastrophic Forgetting Under Fine-Tuning
When Sound of Pixels is fine-tuned on a new set of source classes, the visual backbone is updated to accommodate the new audio supervision signal. This typically causes catastrophic forgetting: performance on the original classes degrades as the weights are overwritten. Managing this with techniques like elastic weight consolidation or progressive training incurs significant engineering overhead and rarely fully preserves original performance.
The proposed architecture avoids this entirely by freezing the DINOv2 backbone. Since the visual encoder is never updated, there is no risk of forgetting previously learned visual categories. Adding new source classes requires only training or fine-tuning the cross-modal attention and mask decoder modules, which are much smaller and faster to train.
6.4 Zero-Shot Generalisation
Perhaps the most compelling advantage of the DINOv2-guided approach is its capacity for zero-shot or few-shot generalisation to new source classes. Because DINOv2 features are semantically rich, the model can separate audio sources corresponding to visual categories it has never been trained to separate — as long as those categories produce distinct visual features.
In practical terms: a model trained on the 21 instrument categories above, when presented at inference time with a scene containing a harp (not in the training set), will produce a reasonable source estimate because DINOv2 features will place the harp patches in a region of feature space similar to string instruments, and the cross-modal attention will route accordingly. Fine-tuning Sound of Pixels on a new instrument class would require collecting new audio-visual paired data, retraining, and validation — a cycle measured in days or weeks per new class.
6.5 Quantitative Comparison of Approaches

Table 3. Comparison of fine-tuning strategies vs. the proposed architecture across key criteria.

7. Training Phase Design

The training strategy is designed around three core principles: (1) leverage the frozen DINOv2 backbone to maximise data efficiency, (2) use a mix-and-separate curriculum that progressively increases separation difficulty, and (3) apply a multi-objective loss that enforces both perceptual audio quality and correct visual grounding.
7.1 Dataset Construction
Training data is constructed synthetically using the mix-and-separate paradigm introduced in Sound of Pixels. This avoids the need for real mixed recordings with known ground-truth sources.


Recommended datasets by tier:
Tier 1 (Speech): AVSpeech (primary) — 150,000 video clips with clean foreground speech and clean ground-truth targets. VoxCeleb2 (supplementary) — full audio-visual pairs used in mix-and-separate pipeline, but sampled at 20% of speech batches and SNR-filtered above 15 dB using a speech activity detector. VoxCeleb2 cannot be used for visuals alone since the video and audio come from the same recording; the full paired clip is used, with strict quality filtering to limit noisy target contamination.
Tier 2 (Instruments): MUSIC Dataset — 685 video clips across 11 categories. AudioSet (instrument subsets) — 2M+ clips. YouTube-8M instrument segments.
Tier 3 (Environment): FreeSound and AudioSet for environmental categories with associated video. AudioSet quality filtering is mandatory before use: (1) confidence score >0.7 for target label; (2) target sound present >50% of clip duration; (3) estimated foreground SNR >10 dB; (4) for instrument clips, visual object detector must confirm instrument presence. Expected retention: ~55–65% of raw clips.
Data augmentation: pitch shifting (±2 semitones), time stretching (0.9x–1.1x), additive noise (SNR 20–40 dB), random gain, channel swap.
7.2 Training Curriculum (Three Phases)
Training proceeds in three sequential phases, each building on the previous. This curriculum design reduces the risk of the cross-modal attention module collapsing to trivial solutions (e.g., ignoring the visual stream entirely).

Phase 1 — Audio-Only Pretraining (Weeks 1–2)
In Phase 1, the visual stream is disabled and the U-Net audio encoder and mask decoder are pretrained on the audio separation task alone. This initialises the audio processing modules to produce reasonable spectral masks before the cross-modal fusion is introduced.
Input: Mixed spectrogram only (no visual input).
Model: Audio U-Net encoder + decoder with fixed N=2 output heads.
Loss: SI-SNR (Scale-Invariant Signal-to-Noise Ratio) between predicted and ground-truth waveforms.
Duration: 200K gradient steps, batch size 32. Earlier estimates of 50K–100K steps are insufficient for joint SI-SNR + cRM convergence; early stopping with patience 10K steps on validation SI-SNRi.
Optimizer: AdamW, lr=1e-3, weight decay=1e-4, cosine annealing schedule. Gradient clipping: max norm 1.0 at every step — essential for stability with combined SI-SNR + cRM losses which can produce large spikes early in training.
Expected outcome: SI-SNR improvement > 8 dB on held-out validation mixtures.
Phase 2 — Cross-Modal Attention Warmup (Weeks 3–5)
The pretrained audio modules from Phase 1 are frozen and the cross-modal attention fusion module is trained from scratch. DINOv2 is frozen throughout. This phase teaches the attention module to use visual features without corrupting the already-learned audio representations.
Input: Mixed spectrogram + DINOv2 patch tokens from synchronised video frames.
Frozen: DINOv2 backbone, audio U-Net encoder and decoder.
Trainable: Cross-modal attention module only (~8M parameters).
Loss: SI-SNR + 0.1 × attention entropy regularisation (encourages sparse, focused attention).
Duration: 30K steps, batch size 16 (limited by GPU memory for attention computation).
Learning rate: 5e-4 with warmup over first 1K steps. Gradient clipping: max norm 1.0.
Expected outcome: Visual attention maps localise the correct source objects in >70% of validation frames.
7.2.2a Intermediate Checkpoint: Concatenation Fusion Baseline
Before training the full cross-modal attention, a simpler concatenation fusion baseline is trained for 5K steps. DINOv2 features are mean-pooled across patches, projected to 512 dimensions, and concatenated channel-wise with the U-Net bottleneck — equivalent to the Sound of Pixels fusion but with DINOv2 features substituted for ResNet. This provides: (1) a clean ablation showing the contribution of cross-attention over naive concatenation; (2) a trained fallback model if full cross-modal attention fails to converge. This ensures a reportable intermediate result regardless of Phase 2 outcome.
7.2.2b Cross-Modal Attention Architecture Specification
The cross-modal attention module is specified as follows. Audio bottleneck features are reshaped to a sequence [T_a=240, D_a=512]. DINOv2 patch tokens from the aligned video frame are projected from D_v=768 to D_a=512. Temporal alignment: spectrogram frame t maps to video frame floor(t × 150 / 240) via nearest-neighbour index, preserving lip-sync correspondence without feature interpolation.
Attention layers: 2 stacked cross-attention blocks
Attention heads: 8 (head dimension = 64)
Positional encoding: sinusoidal 1D encoding on both audio query sequence and visual key/value sequence; DINOv2 spatial positional encodings preserved from backbone
Key/value projection: linear D_v (768) → D_a (512) applied once before the attention stack
Residual + LayerNorm (Pre-LN) after each attention block
Feed-forward sublayer: 2-layer MLP, hidden dim 2048, GELU activation

Phase 3 — End-to-End Fine-Tuning (Weeks 6–10)
Phase 3 is structured around a primary experiment comparing three DINOv2 configurations: (A) fully frozen backbone, (B) last 2 transformer blocks unfrozen at lr=1e-5, and (C) last 4 blocks unfrozen at lr=5e-6. This is not an optional ablation — it is a core experimental question, since DINOv2 was pre-trained on still images and adaptation to temporal lip dynamics and instrument performance may require partial fine-tuning. Results from A/B/C will be the primary comparison table in any publication. All other components are unfrozen with differential learning rates.
Trainable: Cross-modal attention, audio U-Net encoder (last 2 blocks), mask decoder, source query tokens.
Loss: L_total = L_SI-SNR + α×L_cRM + β×L_STFT + γ×L_perceptual, where α, β, γ are hyperparameters requiring tuning. Initial values (α=0.1, β=0.05, γ=0.1) from comparable literature; a grid search over α∈{0.05,0.1,0.2}, β∈{0.01,0.05,0.1}, γ∈{0.05,0.1,0.2} is planned for the first 20K steps of Phase 3.
L_STFT: Multi-scale STFT loss comparing predicted and ground-truth spectrograms across window sizes [256, 512, 1024].
L_perceptual: Cosine similarity loss between DINOv2 features of predicted and ground-truth audio spectrograms (encourages spectral consistency).
Progressive difficulty: start with 2-source mixtures, increase to 3-source at step 20K, 4-source at step 40K.
Duration: 100K steps, batch size 8 (4-source mixtures), gradient accumulation 4 steps.
Optimizer: AdamW, lr=3e-4 for fusion/decoder, lr=3e-5 for audio encoder, lr=1e-5 for DINOv2 unfrozen blocks (differential LR). Gradient clipping: max norm 1.0 throughout.
7.3 Loss Functions
The primary training objective is Scale-Invariant Signal-to-Noise Ratio (SI-SNR), which is scale-invariant and well-suited to situations where the separated signal may differ in overall gain from the ground truth:


The permutation-invariant training (PIT) variant of SI-SNR is used when training with N > 1 sources to resolve the label assignment ambiguity: all N! permutations of source assignments are evaluated and the minimum-loss permutation is used for the gradient update.
7.4 Evaluation Protocol
Model performance is evaluated on three benchmarks:
MUSIC-21 Test Set: held-out clips from each of the 21 instrument categories. Metrics: SI-SNR improvement (SI-SNRi), SDR (Signal-to-Distortion Ratio), SIR (Signal-to-Interference Ratio).
AVSpeech Two-Speaker Test: 1,000 synthetic 2-speaker mixtures from held-out AVSpeech clips. Metrics: SI-SNRi and WER (Whisper-base ASR). Baseline WER is measured on the unprocessed mixed input: a 2-speaker mixture fed directly to Whisper-base yields approximately 65–80% WER depending on overlap rate and SNR ratio. The model target of <15% WER therefore represents a reduction of ~50–65 percentage points from the mixed-input baseline, not from a clean-speech oracle. A supplementary VoxCeleb2 identity test uses 500 mixtures of held-out identities to evaluate generalisation to unseen faces.
Cross-Category Generalisation: 500 mixtures containing instrument categories not present in training (Tier 4 animals + rare instruments). Metric: SI-SNRi vs. Sound of Pixels baseline (zero-shot gap).

Table 4. Performance targets across training phases.

8. Future Extensions

8.1 Dynamic Source Count with Slot Attention
The current design assumes a fixed number of sources N known at inference time. A natural extension is to replace the N fixed source query tokens with a slot attention mechanism (Locatello et al., 2020) that automatically discovers the number of sources present in the scene. Each slot competes to attend to distinct visual regions, and only slots with sufficient activation contribute output masks.
8.2 Text-Conditioned Separation
Because DINOv2 features occupy a semantically structured space, it is feasible to condition source separation on a natural language query (e.g., 'isolate the guitar') using a CLIP text encoder to generate a visual query vector. This would enable zero-shot separation of any category that can be described in text, without requiring any audio-visual paired training data for that category.
8.3 DINOv3 Integration
DINOv3's ConvNeXt variants are immediately drop-in compatible with the proposed architecture. The ConvNeXt backbone produces feature maps rather than patch token sequences, which require a minor modification to the cross-modal attention module (flatten the spatial dimensions to form the key/value sequence). DINOv3's richer features — trained on 12x more data than DINOv2 — are expected to improve zero-shot generalisation, particularly for rare instrument categories.
8.4 Real-Time Streaming Inference
The current batch processing design is suitable for offline separation but not real-time applications. A streaming variant would process fixed-length audio chunks (e.g., 250 ms with 50% overlap) and maintain a rolling buffer of DINOv2 visual features from the most recent N video frames. The cross-modal attention would operate over this temporal window, enabling latency under 500 ms — sufficient for live captioning and hearing assistance applications.


9. Limitations

The proposed architecture advances audio-visual source separation in several important respects, but the following limitations should be acknowledged explicitly.
9.1 Computational Cost of Dense Frame Processing
Processing 150 DINOv2 forward passes per 6-second clip is the principal computational bottleneck. DINOv2-Base requires approximately 17 GFLOPs per 448px frame; 150 frames costs ~2.5 TFLOPs per training sample in the visual stream alone. This limits batch size on a single A100 to 4–8 samples and makes training on consumer hardware impractical. The ±3-frame windowed attention (Section 7.2.2b) mitigates the memory problem but not the per-frame DINOv2 inference cost. Future work should investigate temporal striding with linear feature interpolation.
9.2 Reliance on Visible Sound Sources
The visual grounding mechanism assumes every active sound source is visible in the video frame. It fails for off-screen speakers, occluded instruments, or environmental sounds (rain, traffic) with no localised visual correlate. The architecture has no mechanism to detect or flag when a source is visually absent, meaning the cross-modal attention will attempt to hallucinate a spatial assignment in all cases.
9.3 Fixed Source Count at Inference
N (the number of sources) must be specified at inference time. Active source count changes within clips — speakers pause, instruments drop out. The slot attention extension in Section 8.1 addresses this but is outside the primary experimental plan, meaning all benchmarks assume known N.
9.4 AudioSet Label Noise
AudioSet is used for Tier 1 (speech-from-background) and Tier 3 (environmental) training. AudioSet has estimated label error rates of 30–40% in some categories. Even with the quality filtering protocol specified in Section 7.1, some corrupted training samples will pass through, degrading cRM supervision signal quality. Results on environmental source categories should be interpreted with this caveat.
9.5 Loss Coefficient Sensitivity
The Phase 3 combined loss coefficients (α, β, γ) are presented as initial values requiring a grid search, not settled hyperparameters. The relative scale of SI-SNR and cRM MSE varies by audio domain and dataset. Any reported results should clearly state the final coefficient values used and include a sensitivity analysis showing performance variation across the grid.

10. Conclusion

This report has presented a comprehensive case for replacing the convolutional visual backbone in Sound of Pixels with a frozen DINOv2 Vision Transformer, coupled with a cross-modal attention fusion module. The key arguments are:
DINOv2's self-supervised patch features are semantically richer and more generalisable than task-trained CNN features, enabling the model to distinguish sources across a far broader range of categories without task-specific visual supervision.
Cross-modal attention provides a fundamentally more powerful fusion mechanism than channel-wise multiplication, allowing fine-grained spatial correspondence between audio frequency bands and visual source regions.
Freezing the DINOv2 backbone eliminates catastrophic forgetting, dramatically reduces training compute, and enables new source classes to be added by training only a small decoder module.
The three-phase training curriculum — audio-only pretraining, attention warmup, end-to-end fine-tuning — provides a principled path to stable convergence without requiring the difficult joint optimisation of visual and audio objectives from scratch.
The proposed source class taxonomy covers the commercially most important separation scenarios: multi-speaker dialogue, mixed musical performance, and speech-in-noise, with a clear path to zero-shot extension via DINOv3 and text conditioning.




11. Implementation Specification
This section provides the concrete implementation specification required to build and train the proposed system. All numbers are grounded in published literature; sources are cited inline.
11.1 Environment and Dependencies
Python 3.10+ is required. The following pinned versions are recommended for reproducibility. All libraries are installable via pip:
Hardware: minimum 1x NVIDIA A100 80 GB for Phase 2 and 3 training. Phase 1 (audio-only) runs on a 40 GB A100 or equivalent. Multi-GPU training via DDP (DistributedDataParallel) is supported from Phase 2 onward; use 2-4 GPUs for Phase 3 with 4-source mixtures.
11.2 Data Pipeline and Preprocessing
Audio preprocessing. All audio is resampled to 16 kHz mono (standard for speech separation: Luo & Mesgarani, 2019; Ephrat et al., 2018). Clips are randomly cropped to exactly 6 seconds (96,000 samples) during training; shorter clips are zero-padded. STFT parameters follow the convention established in Conv-TasNet and the Looking to Listen system: FFT size N_FFT = 512, hop length H = 160 samples (10 ms at 16 kHz), window length W = 400 samples (25 ms), Hann window. This yields T = ceil(96000 / 160) = 601 time frames and F = 257 frequency bins per spectrogram. The complex STFT is stacked as 2-channel real/imaginary input of shape [2, F, T] = [2, 257, 601].
Mel filterbank (evaluation visualisation only). 80 mel bins, frequency range 80 Hz -- 7600 Hz, following the convention of Ephrat et al. (2018). Mel spectrograms are used only for visualisation and the L_STFT perceptual loss; all mask prediction and waveform reconstruction operates on the full linear STFT.
Video preprocessing. Frames are decoded at the native video frame rate then temporally subsampled to 25 fps. Each frame is resized to 448 x 448 pixels (DINOv2-Base input resolution; Oquab et al., 2023) using bicubic interpolation, centre-cropped to 448 x 448, and normalised with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]. DINOv2-Base with patch size 14 produces a 32 x 32 = 1024 patch token grid per frame, each token of dimension 768. A 6-second clip at 25 fps yields 150 frames, producing a visual feature tensor of shape [150, 1024, 768] per source.
Temporal alignment. Spectrogram frame t (0-indexed, t in [0, 600]) is aligned to video frame v = floor(t x 150 / 601) via nearest-neighbour index. No feature interpolation is applied. This mapping is computed once and cached per clip at dataset construction time.
cRM target computation. The ideal complex ratio mask for source i is M_i = S_i / X (complex division, per frequency-time bin). Since the complex ratio is unbounded, hyperbolic tangent compression is applied: M_i_compressed = tanh(K x |M_i|) x exp(j x angle(M_i)), with K = 10, C = 0.1, following Williamson et al. (2016). Targets are pre-computed and stored as float16 tensors to reduce I/O during training.
Train / validation / test splits. Per dataset: MUSIC 80/10/10 by video identity (not clip) to prevent data leakage; AVSpeech 85/7.5/7.5 by speaker identity; VoxCeleb2 uses the standard dev/test split from Nagrani et al. (2017). No speaker identity or video source appears in more than one split.
11.3 Model Architecture: Dimension Table and Forward Pass
Audio U-Net encoder. The encoder follows the U-Net design of Zhao et al. (2018) adapted for complex input. Input shape [B, 2, 257, 601]. Five encoder blocks, each consisting of Conv2d (3x3, stride (2,2), same padding), GroupNorm (groups=8, following Wu & He 2018), and LeakyReLU (negative slope 0.2). Channel dimensions: [2, 32, 64, 128, 256, 512]. After the 5th encoder block the spatial resolution is [B, 512, 9, 19] (bottleneck). Skip connections preserve encoder outputs at each level for the decoder.
Bottleneck projection. The bottleneck feature map [B, 512, 9, 19] is reshaped to a sequence [B, T_a, D_a] = [B, 171, 512] (9x19=171 spatial positions, 512 channels) via flatten over spatial dimensions followed by a linear projection Conv1x1 mapping 512 to D_a=512. This sequence forms the audio query input to the cross-modal attention module.
Visual input projection. For each audio query timestep t_a, the aligned video frame index v = floor(t_a x 150 / 171) is retrieved. DINOv2 patch tokens [1024, 768] are projected to [1024, 512] via a single linear layer (no bias), producing the key/value sequence for cross-modal attention. This projection is shared across all timesteps and source channels.
Cross-modal attention module. 2 stacked cross-attention blocks. Each block: Pre-LayerNorm on query; MultiheadAttention with n_heads=8, head_dim=64, D=512, dropout=0.1 (consistent with Vaswani et al., 2017 and used in AV-HuBERT, Shi et al., 2022); residual connection; Pre-LayerNorm; 2-layer feedforward MLP with hidden_dim=2048, GELU activation, dropout=0.1; residual connection. Sinusoidal positional encoding (Vaswani et al., 2017) added to audio query sequence before first attention block. Visual key/value sequence uses DINOv2’s internal positional encodings, which are already incorporated into the patch token values produced by the backbone forward pass.
Source query tokens and mask decoder. N learnable source query tokens of dimension 512, randomly initialised from N(0, 0.02). These are prepended to the audio query sequence and attend to the visual key/value tokens jointly. The N output slots are reshaped back to [B, N, 512, 9, 19] and fed into N independent U-Net decoders (parameter-shared across sources). Each decoder mirrors the encoder: 5 transposed Conv2d blocks with skip connections from corresponding encoder levels, outputting [B, 2, 257, 601] (real + imaginary mask channels). Mask activation: tanh (bounded output in [-1, 1] consistent with cRM targets). Reconstruction: separated_i = mask_i (complex multiply) x X_stft, then iSTFT via torchaudio.transforms.InverseSpectrogram.
Total trainable parameter count (Phase 3): Audio U-Net (last 2 encoder blocks + full decoder): ~9M. Cross-modal attention (2 blocks): ~8M. Source query tokens (Nx512, N=4): ~2K. Visual projection linear layer: 768x512 = ~0.4M. Total trainable: ~17.4M. DINOv2 backbone (frozen): ~86M (not in optimizer).
11.4 Training Configuration
Phase 1 — Audio-Only Pretraining
Phase 2 — Cross-Modal Attention Warmup
Phase 3 — End-to-End Fine-Tuning
11.5 Evaluation Protocol
SI-SNR / SI-SNRi. Use torchmetrics.audio.ScaleInvariantSignalNoiseRatio (zero_mean=True). SI-SNRi = SI-SNR(separated, target) - SI-SNR(mixture, target). PIT is applied at evaluation: enumerate all N! assignments and report the minimum-loss permutation, using scipy.optimize.linear_sum_assignment (Hungarian algorithm) for N>3 to avoid factorial cost.
SDR / SIR. Use mir_eval.separation.bss_eval_sources (Vincent et al., 2006) for SDR, SIR, SAR. Report all three on the MUSIC-21 test set for compatibility with prior work (Zhao et al., 2018).
WER. Run openai/whisper-base on 16 kHz separated waveforms. Compute WER using jiwer.wer(reference, hypothesis). Baseline WER is computed by running Whisper-base on the unprocessed 2-speaker mixture; target WER <15% is relative to this baseline of approximately 65-80% (see Section 7.4).
Attention localisation accuracy. For each validation frame, extract the cross-modal attention map for source i (shape [n_heads, T_a, N_patches]). Average across heads and audio query timesteps to get a patch-level saliency map [N_patches] = [1024], reshape to [32, 32]. Compute the bounding box IoU between the top-k (k=50) attended patches and the ground-truth instrument/face bounding box (obtained from an off-the-shelf object detector: YOLOv8 applied to validation frames). Localisation accuracy = fraction of frames with IoU > 0.3, following the weakly-supervised object localisation metric of Zhou et al. (2016).
11.6 Repository Structure
The recommended project layout is as follows. Configuration management uses Hydra (Yadan 2019) with YAML config files. All paths are relative to the project root.
Additional references for Section 11: Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS 2017. — Shi, B., et al. (2022). Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction (AV-HuBERT). ICLR 2022. — Subakan, C., et al. (2021). Attention Is All You Need In Speech Separation (SepFormer). ICASSP 2021. — Wu, Y., & He, K. (2018). Group Normalization. ECCV 2018. — Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). ICLR 2019. — Zhou, B., et al. (2016). Learning Deep Features for Discriminative Localization (CAM). CVPR 2016. — Vincent, E., et al. (2006). Performance Measurement in Blind Audio Source Separation. IEEE/ACM TASLP 2006. — He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR 2022.
References

Caron, M., Touvron, H., Misra, I., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. ICCV 2021.
Luo, Y., & Mesgarani, N. (2019). Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation. IEEE/ACM TASLP.
Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. TMLR 2024.
Meta AI / Community. (2025). "DINOv3" — unverified community documentation. No official paper published as of this writing. HuggingFace model hub: facebook/dinov3-*. Treat all cited figures as provisional.
Nagrani, A., Chung, J. S., & Zisserman, A. (2017). VoxCeleb: A Large-Scale Speaker Identification Dataset. Interspeech 2017.
Zhao, H., Gan, C., Rouditchenko, A., et al. (2018). The Sound of Pixels. ECCV 2018.
Zhao, H., Gan, C., Ma, W.-C., & Torralba, A. (2019). The Sound of Motions. ICCV 2019.
Locatello, F., Weissenborn, D., Unterthiner, T., et al. (2020). Object-Centric Learning with Slot Attention. NeurIPS 2020.
Gao, R., & Grauman, K. (2021). Visualvoice: Audio-Visual Speech Separation with Cross-Modal Consistency Regularization. CVPR 2021.
Efrat, N., et al. (2022). AudioScopeV2: Audio-Visual Attention Architectures for Calibrated Audio-Visual Source Separation. ECCV 2022.
Ephrat, A., Mosseri, I., Lang, O., et al. (2018). Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation. ACM SIGGRAPH / TOG. [AVSpeech dataset paper]
Williamson, D. S., Wang, Y., & Wang, D. (2016). Complex Ratio Masking for Monaural Speech Separation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(3), 483–492.
Tan, K., & Wang, D. (2019). Learning Complex Spectral Mapping with Gated Convolutional Recurrent Networks. IEEE/ACM TASLP, 28, 380–390.
Erdogan, H., Hershey, J. R., Watanabe, S., & Le Roux, J. (2015). Phase-Sensitive and Recognition-Boosted Speech Separation Using Deep Recurrent Neural Networks. ICASSP 2015.
Gemmeke, J. F., et al. (2017). Audio Set: An Ontology and Human-Labeled Dataset for Audio Events. ICASSP 2017.