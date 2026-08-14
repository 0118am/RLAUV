"""Rigid-body and hydrodynamic regression cases."""

import pytest

from tests.case_catalog import cases_for


@pytest.mark.parametrize("case", cases_for("physics"), ids=lambda case: case.__name__)
def test_physics_case(case):
    case()
