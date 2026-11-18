"""Shipping costs of an online shop: the Python example of Testuary's
quickstart."""


def shipping_cost(weight_kg: float, express: bool = False, country: str = "DE") -> float:
    """The shipping cost of a parcel in euros."""
    if weight_kg <= 0:
        raise ValueError("the weight must be positive")
    if weight_kg <= 2:
        cost = 4.99
    elif weight_kg <= 10:
        cost = 7.99
    else:
        cost = 7.99 + 0.5 * (weight_kg - 10)
    if country != "DE":
        cost += 6.0
    if express:
        cost *= 1.5
    return round(cost, 2)


def free_shipping(order_total: float, member: bool = False) -> bool:
    """Whether an order ships free of charge."""
    threshold = 30.0 if member else 50.0
    return order_total >= threshold
