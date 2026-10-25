"""CH 227z Spectroscopy Skill Check grader.

A student passes only if their absorbance and concentration estimate are in
their version's ranges and their calculated concentration is within the
tolerance of the Beer's law value from their own absorbance.

Usage:
    python spectroscopy.py                     # grades spectroscopy_input.csv
    python spectroscopy.py Grading_Tracker.xlsx
    python spectroscopy.py --template          # blank input sheet from roster.csv
"""

import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))  # shared grading_core.py lives in CH 227z

from grading_core import AllPass, Computed, RangeCheck, ToleranceCheck, parity_version, run  # noqa: E402


def beers_law_concentration(values, config):
    """c = (A - b) / m from the standard curve A = m*c + b."""
    return (values["absorbance"] - config["intercept"]) / config["slope"]


SPECTROSCOPY = {
    "name": "CH 227z Spectroscopy Skill Check",
    "slug": "spectroscopy",
    "sheet": "Grading",
    "version": parity_version,
    "inputs": {
        "absorbance": "Absorbance",
        "estimate": "Estimate",
        "calculation": "Student Calculation",
    },
    "ranges": {
        "A": {"absorbance": (0.20, 0.45), "estimate": (1.00e-5, 2.00e-5)},
        "B": {"absorbance": (0.50, 0.80), "estimate": (2.25e-5, 3.50e-5)},
    },
    "slope": 22400,       # M^-1
    "intercept": -0.00410,
    "tolerance": 0.05,    # relative; chosen while building the sheet, not set by the course
    "steps": [
        RangeCheck("absorbance_ok", "Within Bounds", "absorbance"),
        RangeCheck("estimate_ok", "Within Bounds", "estimate"),
        Computed("expected", "Expected Calculation", beers_law_concentration,
                 requires=["absorbance"]),
        ToleranceCheck("correct", "Correct?", "calculation", expected="expected"),
        AllPass("pass", "Pass?", ["absorbance_ok", "estimate_ok", "correct"]),
    ],
    # Workbook columns D-K, after Name, Number and Version.
    "columns": ["absorbance", "absorbance_ok", "estimate", "estimate_ok",
                "expected", "calculation", "correct", "pass"],
    "pass": "pass",
}


if __name__ == "__main__":
    run(SPECTROSCOPY, FOLDER)
