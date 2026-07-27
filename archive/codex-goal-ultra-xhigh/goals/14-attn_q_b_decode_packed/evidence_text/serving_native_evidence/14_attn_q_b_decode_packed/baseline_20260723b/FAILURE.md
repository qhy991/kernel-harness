# Baseline/profile bundle `20260723b`

The B200 environment check, both M-bucket reachability traces, six identity
paired series, the frozen f32-scale task, and both Nsys collections completed on
the wrapper-selected GPU recorded in `gpu_identity.csv`.

The bundle then stopped before NCU collection. `nsys stats` rejected its
just-created M=32 SQLite export because the report timestamp was slightly newer
and `--force-export=true` had not been supplied. The full error is preserved in
`bundle.log`.

This directory is useful preflight evidence but is not a complete baseline
profile campaign. The script now forces each Nsys export refresh, and the
complete campaign uses a fresh non-overwriting `20260723c` directory.
