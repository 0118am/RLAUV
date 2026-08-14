"""Frame alignment, replay, and validation regression cases."""

import pytest

from tests.case_catalog import cases_for


@pytest.mark.parametrize("case", cases_for("validation"), ids=lambda case: case.__name__)
def test_validation_case(case):
    case()
