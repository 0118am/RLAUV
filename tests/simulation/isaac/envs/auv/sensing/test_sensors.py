"""Policy-observation transport regression cases."""

import pytest

from tests.case_catalog import cases_for


@pytest.mark.parametrize("case", cases_for("sensing"), ids=lambda case: case.__name__)
def test_sensing_case(case):
    case()
