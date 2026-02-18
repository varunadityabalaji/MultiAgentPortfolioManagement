"""
main.py
CLI entry point for the stock sentiment multi-agent framework.

Usage:
    python main.py --ticker AAPL
    python main.py --ticker TSLA --output ./results/
"""
import argparse
import json
import logging
import os
import sys

# Ensure project root is on path when running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator_agent import OrchestratorAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="Stock Sentiment Multi-Agent Framework"
    )
    parser.add_argument(
        "--ticker", "-t", required=True, help="Stock ticker symbol (e.g. AAPL)"
    )
    parser.add_argument(
        "--output", "-o", default="./output",
        help="Directory to save JSON report (default: ./output)"
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    orchestrator = OrchestratorAgent()

    print(f"\n🔍 Analyzing sentiment for {ticker}...\n")
    report = orchestrator.run(ticker)

    # Print to console
    print(json.dumps(report, indent=2))

    # Save to file
    os.makedirs(args.output, exist_ok=True)
    timestamp = report["timestamp"].replace(":", "-").replace("+", "_")
    filename = f"{ticker}_{timestamp}.json"
    filepath = os.path.join(args.output, filename)
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Report saved to: {filepath}")
    print(
        f"   Sentiment: {report['sentiment_label']}  |  "
        f"Score: {report['sentiment_score']}  |  "
        f"Confidence: {report['confidence']}"
    )


if __name__ == "__main__":
    main()
