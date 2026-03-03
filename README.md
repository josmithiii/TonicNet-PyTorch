# TonicNet-PyTorch

PyTorch reimplementation of **TonicNet** -- a polyphonic music model for 4-part Bach chorale generation.

Based on the paper [Improving Polyphonic Music Models with Feature-Rich Encoding](https://arxiv.org/abs/1911.11775) (Peracha, 2019), originally ported from the [TF2 reimplementation](https://github.com/AI-Guru/tonicnet) by Tristan Behrens, now evolved to a Transformer architecture.

## Architecture

- **Vocabulary**: 99 tokens (song\_start, song\_end, 50 chords, 47 pitches)
- **Embeddings**: token (100d) + repetition (32d) + position (sinusoidal) + bars-remaining countdown
- **4-layer pre-norm causal Transformer** with KV-cache for fast generation
- **GRU alternative**: 3-layer GRU with compressed hidden state (via `GRUTonicNet`)
- **Skip connections**: output concatenated with repetition/position embeddings
- **Voice masking**: enforces valid chord/soprano/bass/alto/tenor token ordering

Sequences interleave 5 tokens per timestep: \[chord, soprano, bass, alto, tenor\] at 16th-note resolution.

## Quick Start

```bash
make setup      # create venv and install dependencies
make generate   # generate 3 sample MIDI files
make wav        # render sample_1.mid to WAV (requires: brew install fluid-synth)
```

Or without Make:

```bash
pip install torch music21 note-seq h5py numpy
python generate.py 3 --weights tonicnet-best.pt
```

## Make Targets

```
make help           Show all targets
make setup          Create venv and install dependencies (uses uv)
make generate       Generate 3 samples (16 bars each)
make train          Fine-tune from existing weights (150 epochs)
make train-scratch  Train from scratch (150 epochs)
make snapshot       Snapshot tonicnet-best.pt with timestamp
make wav            Render sample_1.mid to WAV (requires fluid-synth)
make mp3            Render sample_1.mid to MP3 (requires ffmpeg)
make playmidi       Play sample_1.mid via fluidsynth
make clean          Remove generated MIDI and WAV files
make distclean      Remove venv and all generated files
```

## Scripts

| Script | Purpose |
|--------|---------|
| `model.py` | Transformer and GRU model definitions, vocabulary, voice masking |
| `model_v3.py` | Legacy v3 (pre-countdown) model definition |
| `generate.py` | Autoregressive sampling with MIDI output, chord biasing, soprano seeding |
| `generate_v3.py` | Legacy v3 model generator |
| `train.py` | Training loop with masked loss and CSV logging |

### Generate

```bash
python generate.py [n_samples] [--weights PATH] [--temperature T] [--bars N] [--gpu]
python generate.py 1 --seed soprano.mid --chords chords.txt --chord-bias 2.0
```

Produces MIDI files with random tempo (65-85 QPM) and temperature (0.25-0.75 if not fixed). Supports soprano-seeded harmonization with optional chord constraints and chord-tone logit biasing.

### Train

```bash
python train.py                                                    # train from scratch
python train.py --weights tonicnet-best.pt --epochs 150            # fine-tune
python train.py --model-type gru --out gru-best.pt                 # train GRU variant
python train.py --overwrite --out tonicnet-best.pt                 # overwrite checkpoint
```

Expects `dataset_train.p`, `dataset_valid.p`, `dataset_test.p` (TF2-format pickle files) in the working directory.

## Dependencies

- Python 3.10+
- PyTorch
- music21
- note-seq
- h5py
- NumPy

## Credits

- Original paper and model: [Omar Peracha](https://github.com/omarperacha/TonicNet)
- TF2 reimplementation and pretrained weights: [Tristan Behrens / AI-Guru](https://github.com/AI-Guru/tonicnet)

## License

See the original repositories for license terms.
