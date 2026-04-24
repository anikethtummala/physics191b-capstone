# Surface Code Lab Instructions

## 1. Start The Lab

Open two terminals from the project root:

```bash
cd "/Users/chris/Desktop/quantum-2 project"
```

Terminal 1 starts the API:

```bash
npm run api
```

Terminal 2 starts the web app:

```bash
npm run dev
```

Then open:

```text
http://127.0.0.1:5173/
```

The API runs at:

```text
http://127.0.0.1:8000
```

## 2. Use The Interface

- Choose `memory X` or `memory Z` in the Code section.
- Select benchmark distances with `d=3`, `d=5`, and `d=7`.
- Use Preview distance to change the lattice shown in the main panel.
- Adjust `Depolarizing p` to set the circuit-level noise rate.
- Set `Shots` for the number of samples per distance.
- Set `Seed` for reproducible runs.
- Toggle decoders:
  - `MWPM` runs the real PyMatching baseline.
  - `CNN`, `GNN`, and `Transformer` are benchmark interfaces only in this version. They report checkpoint-required until trained model inference code and compatible model files are added.
- Click `New sample` to draw a fresh syndrome frame.
- Click `Run benchmark` to start an interactive sweep.

## 3. Decoder Implementation Status

`MWPM` is implemented already. It uses Stim to generate and sample rotated surface-code circuits, then uses PyMatching to decode the detector events and calculate logical error rate, runtime, and memory use.

`CNN`, `GNN`, and `Transformer` are not implemented as working neural decoders yet. The API has adapter classes and benchmark result fields for them, but those adapters currently return `checkpoint_required` instead of running inference.

Importing a pre-trained model is not enough by itself unless the model matches a loader/inference adapter in the backend. The next implementation step is to add a concrete neural decoder format, for example PyTorch checkpoints plus preprocessing from detector-event tensors into each architecture's expected input shape.

## 4. Read The Output

- The lattice panel shows data qubits, stabilizer checks, active syndrome detections, and MWPM match lines.
- The Decoder status panel shows whether each decoder completed, needs a checkpoint, or hit an error.
- The Distance sweep chart plots logical error rate against code distance for completed decoders.
- The Measurements table lists logical error rate, runtime per shot, memory use, and status per decoder/distance row.

## 5. Run Checks

Backend tests:

```bash
npm run test:api
```

Frontend tests:

```bash
npm run test:web
```

Production frontend build:

```bash
npm run build
```

## 6. Useful API Calls

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Fetch a Surface-17 layout:

```bash
curl "http://127.0.0.1:8000/api/layout?distance=3&basis=x"
```

Create a sample syndrome:

```bash
curl -X POST http://127.0.0.1:8000/api/sample \
  -H "Content-Type: application/json" \
  -d '{"distance":3,"basis":"x","noise":{"p":0.001},"seed":1337}'
```

Start a small MWPM benchmark:

```bash
curl -X POST http://127.0.0.1:8000/api/benchmarks \
  -H "Content-Type: application/json" \
  -d '{"distances":[3],"basis":"x","noise":{"p":0.001},"shots":100,"seed":1337,"decoders":["mwpm"]}'
```

Use the returned `job_id` to poll:

```bash
curl http://127.0.0.1:8000/api/benchmarks/YOUR_JOB_ID
```

## 7. Troubleshooting

- If the web app cannot reach the API, confirm `npm run api` is still running on port `8000`.
- If `quantum_stack` is false in `/api/health`, reinstall the Python dependencies:

```bash
.venv/bin/pip install -r services/api/requirements.txt
```

- If port `5173` or `8000` is already in use, stop the existing process or change the port in the relevant command.
- Neural decoders intentionally require additional implementation; this first version only includes their benchmark interfaces.
