"""Entry point for QuantSchedule13DandG.

Run directly::

    python main.py search "Berkshire Hathaway" --max-results 5
    python main.py download 1067983 0001193125-24-123456 --parse --save-db
    python main.py parse filings/0001193125-24-123456.htm --save-db
    python main.py analyze --top 10 --chart

Or after installing with pip::

    edgar search "Apple" --start-date 2023-01-01
"""

from edgar.cli import main

if __name__ == "__main__":
    main()
