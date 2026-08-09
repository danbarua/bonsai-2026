"""Guarded entry point for the Stage 2B amendment-impact audit."""
import argparse
import json

from stage2b_audit import require_audit_prerequisites


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoded-150", required=True,
                        help="validated 150-step encoded artifact")
    parser.add_argument("--production-ridge-rerun", required=True,
                        help="validated 1,200-step thirteen-grid ridge artifact")
    parser.add_argument("--preflight", action="store_true",
                        help="check the sequencing gate and print its status")
    args = parser.parse_args(argv)
    require_audit_prerequisites(args.encoded_150, args.production_ridge_rerun)
    if args.preflight:
        print(json.dumps({"sequencing_gate": "open", "audit": "ready"}, indent=2))
        return 0
    raise RuntimeError(
        "prerequisites are present, but validated GCS composition is not yet "
        "configured; use --preflight until artifact paths are wired")


if __name__ == "__main__":
    raise SystemExit(main())
