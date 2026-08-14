"""Water-current, boundary, surface, and sloshing regression cases."""

import pytest

from tests.case_catalog import cases_for


@pytest.mark.parametrize("case", cases_for("environment"), ids=lambda case: case.__name__)
def test_environment_case(case):
    case()
