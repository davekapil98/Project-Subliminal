# Legacy v1.2 GB100/Isaac implementation

The repository retains the v1.2 custom Isaac data-generator source solely for
historical review under `archive/v1.2_stage1_gb100/`. Master spec v1.3 removed
paid GB100 rental and mandatory custom simulation generation from the
reproducible baseline.

Consequently:

- the active `launch_stage1.sh` and `.launch_stage1_impl.sh` refuse to run;
- old source, tests, containers, configs, requirements, launchers and documents
  live outside the active package, test and configuration trees;
- the archive is neither installed nor collected by the active pytest baseline;
  and
- no result from those legacy tests is used to claim v1.3 phase completion.

Git history preserves the fully operational v1.2 implementation at commit
`4055186`. The archive is unmaintained and must not be imported or launched in
the v1.3 baseline. It may be recovered on a separate experimental branch if
