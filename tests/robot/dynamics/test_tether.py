"""Optional tether and winch regression cases."""

import pytest

from tests.case_catalog import cases_for


@pytest.mark.parametrize("case", cases_for("tether"), ids=lambda case: case.__name__)
def test_tether_case(case):
    case()
