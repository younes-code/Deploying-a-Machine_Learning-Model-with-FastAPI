# test_main.py

from main import add_numbers

def test_addition():
    result = add_numbers(1, 1)
    assert result == 2
