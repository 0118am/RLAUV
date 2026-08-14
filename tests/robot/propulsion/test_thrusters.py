"""Thruster geometry, mapping, and actuator regression cases."""

import pytest

from tests.case_catalog import cases_for


@pytest.mark.parametrize("case", cases_for("thrusters"), ids=lambda case: case.__name__)
def test_thruster_case(case):
    case()
