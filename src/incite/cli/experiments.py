"""Experiment management commands: experiments {list, compare}."""


def register(subparsers):
    """Register experiments commands."""
    exp_parser = subparsers.add_parser("experiments", help="Manage experiment logs")
    exp_subparsers = exp_parser.add_subparsers(dest="exp_command", help="Experiments subcommands")

    exp_list_parser = exp_subparsers.add_parser("list", help="List recent experiment runs")
    exp_list_parser.add_argument(
        "--limit", "-n", type=int, default=20, help="Number of runs to show"
    )

    exp_compare_parser = exp_subparsers.add_parser("compare", help="Compare experiment runs")
    exp_compare_parser.add_argument("run_ids", nargs="+", help="Run IDs to compare")

    exp_diff_parser = exp_subparsers.add_parser("diff", help="Per-query diff between two runs")
    exp_diff_parser.add_argument("run_a", help="Baseline run ID")
    exp_diff_parser.add_argument("run_b", help="Comparison run ID")
    exp_diff_parser.add_argument(
        "--metric", type=str, default="recall@10", help="Metric to compare (default: recall@10)"
    )
    exp_diff_parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top improved/regressed queries to show"
    )

    exp_delete_parser = exp_subparsers.add_parser("delete", help="Delete an experiment run")
    exp_delete_parser.add_argument("run_id", help="Run ID (or prefix) to delete")
    exp_delete_parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )

    exp_parser.set_defaults(func=cmd_experiments)


def cmd_experiments(args):
    """Manage experiment logs."""
    from incite.evaluation import ExperimentLogger

    logger = ExperimentLogger()

    if args.exp_command == "list":
        runs = logger.list_runs(limit=args.limit)
        if not runs:
            print("No experiment runs found.")
            return

        print(f"{'ID':<10} {'Timestamp':<20} {'Method':<8} {'R@10':<8} {'Notes'}")
        print("-" * 70)
        for run in runs:
            r10 = run.results.get("recall@10", 0)
            notes = (run.notes or "")[:25]
            ts = run.timestamp[:19]
            print(f"{run.id:<10} {ts:<20} {run.config.method:<8} {r10:<8.3f} {notes}")

    elif args.exp_command == "compare":
        print(logger.compare(args.run_ids))

    elif args.exp_command == "diff":
        print(logger.diff_runs(args.run_a, args.run_b, metric=args.metric, top_n=args.top_n))

    elif args.exp_command == "delete":
        run = logger.get_run(args.run_id)
        if not run:
            print(f"No run found matching '{args.run_id}'")
            return
        if not args.yes:
            r10 = run.results.get("recall@10", 0)
            notes = (run.notes or "")[:40]
            print(f"Run:    {run.id}")
            print(f"Date:   {run.timestamp[:19]}")
            print(f"Method: {run.config.method}  R@10: {r10:.3f}")
            if notes:
                print(f"Notes:  {notes}")
            confirm = input("\nDelete this run? [y/N] ")
            if confirm.lower() not in ("y", "yes"):
                print("Cancelled.")
                return
        if logger.delete_run(args.run_id):
            print(f"Deleted run {run.id}")
        else:
            print("Delete failed.")

    else:
        print("Usage: incite experiments {list|compare|diff|delete}")
