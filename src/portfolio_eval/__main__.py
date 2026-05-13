"""CLI entry point: python -m portfolio_eval <subcommand>."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="portfolio_eval",
        description="Synthetic portfolio-strategy evaluation benchmark",
    )
    sub = parser.add_subparsers(dest="command")

    # generate_dataset
    gen = sub.add_parser("generate_dataset", help="Generate synthetic dataset and split into public/hidden")
    gen.add_argument("--config", required=True, help="Path to config YAML")
    gen.add_argument("--public-out", required=True, help="Output directory for public benchmark splits")
    gen.add_argument("--hidden-out", required=True, help="Output directory for hidden benchmark splits")

    # evaluator
    ev = sub.add_parser("evaluator", help="Evaluate a strategy on benchmark data")
    ev.add_argument("--public-data", required=True, help="Path to public benchmark data directory")
    ev.add_argument("--hidden-data", required=True, help="Path to hidden benchmark data directory")
    ev.add_argument("--strategy", required=True, help="Path to strategy .py file")
    ev.add_argument("--out", required=True, help="Path to output results JSON file")

    args = parser.parse_args()

    if args.command == "generate_dataset":
        from portfolio_eval.generate_dataset import main as gen_main
        gen_main(args.config, args.public_out, args.hidden_out)
    elif args.command == "evaluator":
        from portfolio_eval.evaluator import main as eval_main
        eval_main(args.public_data, args.hidden_data, args.strategy, args.out)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
