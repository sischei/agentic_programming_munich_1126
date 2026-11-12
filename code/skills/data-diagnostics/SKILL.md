---
name: data-diagnostics
description: Run data quality checks on a CSV file. Reports shape, types, missing values, summary statistics, outlier flags, and duplicates without modifying the source file.
---

# /data-diagnostics

## Description
Run data quality checks on a CSV file. Reports shape, types, missing values,
summary statistics, outlier flags, and duplicate detection. Outputs a concise
diagnostic report without modifying the source file.

## Instructions

You are a data quality analyst. When the user invokes `/data-diagnostics`,
perform the following steps on the CSV file they specify.

### Step 1: Identify the target file
- If the user provided a file path, use that.
- If not, look for CSV files in `data/` and ask which one to check.
- NEVER modify the source file.

### Step 2: Load and inspect
- Load the CSV with `pandas.read_csv()`.
- Print the following header block:
  ```
  === Data Diagnostics: <filename> ===
  Shape: (rows, cols)
  Memory: X.X MB
  ```

### Step 3: Column-level report
For each column, report:
- **dtype** (int, float, string, datetime)
- **n_missing** and **pct_missing**
- **n_unique** (flag if n_unique == n_rows, likely an ID column)
- For numeric columns: mean, std, min, p25, median, p75, max
- For string columns: top 5 most frequent values with counts
- For datetime columns: min date, max date, gaps

### Step 4: Data quality flags
Check and report:
- [ ] Any column with >20% missing values
- [ ] Any numeric column where max/min ratio > 1000 (possible outliers)
- [ ] Any duplicate rows (report count)
- [ ] Any constant columns (zero variance)
- [ ] Any column name with spaces or special characters
- [ ] Row count sanity: is it suspiciously round (e.g., exactly 1000)?

### Step 5: Output
- Print the full report to stdout in plain text.
- Save a copy to `notes/diagnostics_<filename>_<date>.md`.
- End with a one-paragraph summary of the most important findings.

### Constraints
- Do NOT modify the source CSV file under any circumstances.
- Do NOT drop rows or impute missing values.
- Use pandas for loading; numpy for computations.
- If the file is larger than 100MB, warn the user and sample 10,000 rows.

## Example usage

```
> /data-diagnostics data/firm_panel.csv
```

Expected output:
```
=== Data Diagnostics: firm_panel.csv ===
Shape: (2500, 6)
Memory: 0.1 MB

Column: firm
  dtype: int64 | missing: 0 (0.0%) | unique: 5
  min=1  p25=2  median=3  p75=4  max=5

Column: t
  dtype: int64 | missing: 0 (0.0%) | unique: 500
  min=1  p25=125  median=250  p75=375  max=500

Column: Y
  dtype: float64 | missing: 0 (0.0%) | unique: 2500
  min=1.14  p25=1.61  median=1.72  p75=1.83  max=2.55

...

Quality flags:
  [PASS] No columns with >20% missing
  [PASS] No extreme outliers detected
  [PASS] No duplicate rows
  [PASS] No constant columns
  [PASS] Column names are clean

Summary: Clean balanced panel of 5 economies over 500 periods.
No missing values or quality issues detected. Ready for analysis.
```
