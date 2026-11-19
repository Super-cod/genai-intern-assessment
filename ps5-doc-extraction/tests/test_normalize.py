from datetime import date
from decimal import Decimal

import pytest

from app.pipeline.normalize import parse_amount, parse_currency, parse_date


@pytest.mark.parametrize("raw,expected", [
    ("March 5, 2024", date(2024, 3, 5)), ("5 March 2024", date(2024, 3, 5)), ("2024-03-05", date(2024, 3, 5)),
    ("15-Jan-2024", date(2024, 1, 15)), ("12.02.2024", date(2024, 2, 12)),   # dotted = day-first convention
    ("25/12/2024", date(2024, 12, 25)),                                     # only one valid reading
    ("1st July 2024", date(2024, 7, 1)),
])
def test_dates(raw, expected):
    p = parse_date(raw)
    assert p.ok and not p.ambiguous and p.value == expected


def test_ambiguous_slash_date_is_flagged_not_guessed():
    p = parse_date("03/04/2024")
    assert p.ambiguous and p.value is None and set(p.candidates) == {"2024-04-03", "2024-03-04"}


def test_unparseable_date():
    assert not parse_date("sometime next week").ok


@pytest.mark.parametrize("raw,expected", [
    ("1,234.56", "1234.56"), ("1.234,56", "1234.56"), ("1 234,56", "1234.56"), ("$ 1,200", "1200.00"),
    ("1,25,000.00", "125000.00"), ("465,50", "465.50"), ("USD 99.9", "99.90"), ("Rs. 1,29,800.00", "129800.00"),
])
def test_amounts(raw, expected):
    assert parse_amount(raw).value == Decimal(expected)


def test_amount_without_digits_is_invalid():
    assert not parse_amount("N/A").ok


@pytest.mark.parametrize("raw,expected", [("USD", "USD"), ("€", "EUR"), ("£", "GBP"), ("Rs.", "INR"), ("EUR 10", "EUR")])
def test_currency(raw, expected):
    assert parse_currency(raw).value == expected


def test_bare_dollar_is_ambiguous_unless_code_in_document():
    assert parse_currency("$").ambiguous
    assert parse_currency("$", "Total (CAD): $50").value == "CAD"
