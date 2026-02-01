from film_recoomendations.foo import foo


def test_foo() -> None:
    """Assert that foo returns its input unchanged.

    Returns:
        None
    """
    assert foo("foo") == "foo"
