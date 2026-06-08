# M3Resp Stage 1

Stage 1 creates `m3resp` as a small integration package. It does not merge the
upstream EIT and EMG codebases.

## Package Boundaries

Use `m3resp` for:

- `M3Session`;
- common event models;
- adapters around upstream packages;
- synchronization;
- export;
- multimodal examples.

Keep modality algorithms upstream:

- EIT algorithms in `eitprocessing`;
- EMG algorithms in `resurfemg`.

## Optional Dependencies

The adapters import optional packages lazily. This allows:

```bash
pip install m3resp
```

for session/export work, and:

```bash
pip install "m3resp[all]"
```

when the upstream packages are available.
