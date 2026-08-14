"""Pool dynamics profile schema and application regression cases."""

import pytest

from tests.case_catalog import cases_for


@pytest.mark.parametrize("case", cases_for("profiles"), ids=lambda case: case.__name__)
def test_profile_case(case):
    case()
