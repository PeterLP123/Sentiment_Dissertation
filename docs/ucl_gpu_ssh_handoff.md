# GPU SSH Handoff

We treat the UCL GPU machines as a pool of replaceable workers behind one stable SSH gateway. This is not a Slurm-style cluster: we choose or reserve a workstation, connect to it through `knuckles.cs.ucl.ac.uk`, and run the job directly on that host.

## Connection shape

```text
Mac -> knuckles.cs.ucl.ac.uk (gateway) -> <gpu-host>.cs.ucl.ac.uk (worker)
```

The Mac SSH config holds the reusable details. Keep one gateway alias and give each current GPU machine an alias that uses `ProxyJump`:

```sshconfig
Host ucl-knuckles
    HostName knuckles.cs.ucl.ac.uk
    User <ucl-cs-username>
    IdentityFile ~/.ssh/id_ed25519

Host <gpu-alias>
    HostName <gpu-host>.cs.ucl.ac.uk
    User <ucl-cs-username>
    ProxyJump ucl-knuckles
    ForwardAgent yes
```

Then `ssh <gpu-alias>` performs both hops. Public-key authentication stays on the Mac. `ForwardAgent yes` lets the remote repository use the Mac's GitHub identity without copying a private key onto the GPU host; enable it only for trusted hosts.

## Storage shape

The workstation is compute, not the source of truth. Repositories and durable outputs live under the shared project filesystem, so changing GPU hosts does not move the project:

```text
<project-root>/
  repos/<project>/       Git checkout; code and small configs
  envs/<project>/        Reproducible Python/runtime environment
  data/raw/              Immutable or licensed inputs; not Git
  runs/<project>/        Checkpoints and resumable run outputs; not Git
  artifacts/models/      Model weights/cache; not Git
  artifacts/logs/        Persistent logs; not Git
```

## Normal run sequence

1. Reserve a remote GPU workstation, or check the permitted pool for a free machine.
2. SSH through the gateway and run `nvidia-smi`. Do not start if another compute process is using the GPU.
3. Work from `<project-root>/repos/<project>` and keep machine-specific paths in environment variables or launcher defaults.
4. Start long jobs in named `tmux` sessions, write logs and incremental checkpoints to shared storage, and make reruns resume rather than overwrite.
5. Record the host, commit, environment/package versions, model, parameters, seed, and output path with each run.
6. Commit code and lightweight manifests to Git; leave data, environments, weights, logs, and generated results on shared storage.

Useful checks are `nvidia-smi`, `tmux ls`, `tmux attach -t <session>`, and `git rev-parse HEAD`. The current UCL host inventory and access guidance are published on the [CS GPU page](https://tsg.cs.ucl.ac.uk/gpus/) and [remote GPU workstation page](https://tsg.cs.ucl.ac.uk/remote-gpu-workstations/).
