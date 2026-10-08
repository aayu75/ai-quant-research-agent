from langchain_core.tools import tool


@tool
def calculate_return(start_price: float, end_price: float) -> float:
    """
    Calculate the percentage return between
    a starting price and an ending price.
    """

    if start_price <= 0:
        raise ValueError("Start price must be greater than zero.")

    return ((end_price - start_price) / start_price) * 100