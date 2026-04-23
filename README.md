# QuantSchedule13DandG

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/FanZouGit/QuantSchedule13DandG.git
   cd QuantSchedule13DandG
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Running the App

The app is controlled through a CLI with four sub-commands: `search`, `download`, `parse`, and `analyze`.

**Search for filings**

```bash
python main.py search "Berkshire Hathaway" --max-results 5
python main.py search "Apple" --start-date 2023-01-01
```

**Download a specific filing by CIK and accession number**

```bash
python main.py download 1067983 0001193125-24-123456
# Download, parse immediately, and save to the local database
python main.py download 1067983 0001193125-24-123456 --parse --save-db
```

**Parse a locally saved filing**

```bash
python main.py parse filings/0001193125-24-123456.htm
# Save parsed data to the local database
python main.py parse filings/0001193125-24-123456.htm --save-db
```

**Analyze filings stored in the local database**

```bash
python main.py analyze --top 10
python main.py analyze --top 10 --chart
python main.py analyze --changes
python main.py analyze --issuer "Apple" --chart
```

> **Tip:** Pass `--db <path>` to any command to use a custom SQLite database file (default: `filings.db`).

---

## Development Plan

### 1. Define Project Requirements
- **Objective**: Create a Python app to retrieve and analyze Schedule 13D/13G filings from the SEC's EDGAR system.
- **Core Features**:
  - Download 13D/13G filings based on keywords, company names, or CIK codes.
  - Parse and extract holdings data from the filings.
  - Analyze trends, ownership changes, and provide visual insights.

---

### 2. Plan the Architecture
A modular structure is recommended for clarity and reusability:
- **Modules**:
  1. **Data Retrieval**: Module to interact with the SEC EDGAR API.
  2. **File Parsing**: Extract relevant sections from filings (like holdings or changes).
  3. **Data Analysis**: Include tools for analysis and visualization.
  4. **User Interface**: A CLI or simple UI for interaction.

---

### 3. Implementation Details

#### A. Basic Configuration
- **Requirements**:
  - Python 3.x
  - Libraries: `requests`, `beautifulsoup4`, `lxml`, `pandas`, optional libraries like `matplotlib`.
  - Use `venv` or `poetry` for environment management.

#### B. Data Retrieval: Download EDGAR Filings
- Use the SEC's EDGAR API:
  - Query filings using keywords, company CIK, or forms (`13D`, `13G`).
- Implementation Example (Python):
  ```python
  import requests

  def download_data(cik, form_type):
      url = f"https://www.sec.gov/edgar/data/{cik}"
      headers = {"User-Agent": "YourEmail/ContactInfo"}
      response = requests.get(url, headers=headers)
      if response.status_code == 200:
          return response.text
  ```

#### C. Parse HTML/Extract Key Data
- Use BeautifulSoup to extract sections like ownership percentages:
  ```python
  from bs4 import BeautifulSoup

  def parse_filings(content):
      soup = BeautifulSoup(content, "lxml")
      holdings = soup.find_all("ownership")
      return holdings
  ```

#### D. Analysis Tools
- Use `pandas` to manipulate and analyze data.
- Add visualization with libraries like `matplotlib` or `plotly`.

#### E. User Interface (Optional Extensions)
- Add CLI commands with `argparse`.
- Optional GUI using `tkinter` or `PyQt`.

---

### 4. Scale and Advanced Features
- **Interactivity**:
  - Save filings locally for offline analysis.
  - Integration with a database for scalable storage.
- **Machine Learning** (Optional):
  - Predict trends in holdings or stock performance using filing data.
