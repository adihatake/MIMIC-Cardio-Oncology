"""
generate_hf_cardiotox_table.py — backward-compatible shim.

Delegates to generate_cohort.main() with the "hf_cardiotox" pipeline config.
New code should call generate_cohort.main() directly.
"""

from pathlib import Path
from cohort_src.generate_cohort import main as _main


def main(
    data_location: Path,
    output_name: str | None = None,
    cycle_sql_dir: Path | None = None,
    prescriptions_sql_dir: Path | None = None,
) -> None:
    _main(
        data_location         = data_location,
        pipeline              = "hf_cardiotox",
        output_name           = output_name,
        cycle_sql_dir         = cycle_sql_dir,
        prescriptions_sql_dir = prescriptions_sql_dir,
    )
