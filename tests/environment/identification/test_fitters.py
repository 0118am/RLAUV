"""Synthetic-data fitting and calibration workflow regression cases."""

import pytest

from tests.case_catalog import cases_for


@pytest.mark.parametrize("case", cases_for("identification"), ids=lambda case: case.__name__)
def test_identification_case(case):
    case()
