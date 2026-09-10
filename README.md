# PaliGemma from Scratch

A from-scratch PyTorch implementation of [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224), a multimodal
vision-language model: a SigLIP vision encoder whose image embeddings are merged with text embeddings and fed to a
Gemma language model for conditional generation.

This repository is my reproduction of the code built in the YouTube tutorial
**[Coding a Multimodal (Vision) Language Model from scratch in PyTorch with full explanation](https://www.youtube.com/watch?v=vAmKB7iPkWw)**
by **[Umar Jamil](https://www.youtube.com/@umarjamilai)**. All credit for the material, the explanations, and the
overall design goes to him — I typed it out to learn how the pieces fit together. If you find this useful, go watch
the original video and subscribe to his channel; it is one of the clearest walkthroughs of a modern VLM available.

## Commit history as a tutorial index

**Each commit corresponds to one major step in the video.** The history is meant to be read in order, so you can check
out any commit to see the codebase exactly as it stood at that point in the tutorial:

| # | Commit | Step |
|---|--------|------|
| 1 | `33ff18d` | SigLIP high-level architecture |
| 2 | `d69200d` | Vision embeddings |
| 3 | `7207a35` | SigLIP encoder |
| 4 | `22ec3fa` | Image and text preprocessing |
| 5 | `4c8c75a` | Model configs |
| 6 | `b89d77d` | Merging image and text inputs |
| 7 | `54acbfa` | Wrapper classes |
| 8 | `842aad3` | RMSNorm and the Gemma MLP |
| 9 | `386ff7a` | Gemma attention: grouped-query / multi-query attention |
| 10 | `be53680` | KV cache |
| 11 | `71f422e` | Rotary position embeddings (RoPE) |
| 12 | `9e94ab4` | `inference.py` main function |
| 13 | `dc6a961` | `utils.py` |
| 14 | `3d82c9f` | Helper functions |
| 15 | `2a44f9a` | Minor cleanup |
| 16 | `309fde1` | Shell command to run inference |

Browse it yourself with:

```bash
git log --oneline --reverse
```

## Layout

| File | Contents |
|------|----------|
| `modeling_siglip.py` | SigLIP vision tower — patch embeddings, encoder layers, self-attention, MLP |
| `modeling_gemma.py` | Gemma language model, KV cache, RoPE, GQA attention, and the `PaliGemmaForConditionalGeneration` wrapper that merges image and text features |
| `processing_paligemma.py` | `PaliGemmaProcessor` — image resizing/rescaling/normalization and prompt construction with image tokens |
| `utils.py` | Loads HF weights from `*.safetensors` plus `config.json` into the model |
| `inference.py` | Generation loop with top-p sampling |
| `launch_inference.sh` | Example invocation with all the knobs set |

## Requirements

```bash
pip install torch transformers safetensors pillow numpy fire
```

## Running it

Download the pretrained weights, e.g.:

```bash
huggingface-cli download google/paligemma-3b-pt-224 --local-dir ~/projects/paligemma-weights/paligemma-3b-pt-224
```

(PaliGemma is a gated model — accept the license on the Hugging Face model page and log in with
`huggingface-cli login` first.)

Then edit the paths at the top of `launch_inference.sh` — `MODEL_PATH`, and `IMAGE_FILE_PATH` pointing at an image of
your own — and run:

```bash
bash launch_inference.sh
```

Or call the script directly:

```bash
python inference.py \
    --model_path "$HOME/projects/paligemma-weights/paligemma-3b-pt-224" \
    --prompt "this building is " \
    --image_file_path "test_images/pic1.jpeg" \
    --max_tokens_to_generate 100 \
    --temperature 0.8 \
    --top_p 0.9 \
    --do_sample False \
    --only_cpu False
```

## Acknowledgements

- [Umar Jamil](https://www.youtube.com/@umarjamilai) for the tutorial this repository reproduces.
- Google DeepMind for PaliGemma, SigLIP, and Gemma.
