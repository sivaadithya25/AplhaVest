from langchain_core.tools import tool

@tool
def calculate_cagr(beginning: float, ending: float, years: float) -> str:
    """Calculate CAGR. Requires beginning value, ending value and years."""
    if beginning <= 0 or ending < 0 or years <= 0:
        raise ValueError("Beginning > 0, ending >= 0, years > 0 are required.")
    value = ((ending / beginning) ** (1 / years) - 1) * 100
    return f"{value:.2f}%"

@tool
def calculate_growth(old_value: float, new_value: float) -> str:
    """Calculate percentage growth from an old value to a new value."""
    if old_value == 0:
        raise ValueError("Old value cannot be zero.")
    value = ((new_value - old_value) / abs(old_value)) * 100
    return f"{value:.2f}%"

@tool
def calculate_roi(initial_value: float, final_value: float) -> str:
    """Calculate simple ROI percentage."""
    if initial_value == 0:
        raise ValueError("Initial value cannot be zero.")
    value = ((final_value - initial_value) / abs(initial_value)) * 100
    return f"{value:.2f}%"

@tool
def compare_returns(first_return: float, second_return: float) -> str:
    """Compare two percentage returns and state the difference."""
    diff = first_return - second_return
    if diff > 0:
        return f"First return is higher by {diff:.2f} percentage points."
    if diff < 0:
        return f"Second return is higher by {abs(diff):.2f} percentage points."
    return "Both returns are equal."

@tool
def create_comparison_table(items: str) -> str:
    """Turn lines formatted as Company|Metric1|Metric2 into a simple Markdown table."""
    rows = [x.strip() for x in items.splitlines() if x.strip()]
    if not rows:
        return "No rows supplied."
    parsed = [r.split("|") for r in rows]
    width = max(len(r) for r in parsed)
    parsed = [r + [""] * (width - len(r)) for r in parsed]
    header = parsed[0]
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * width) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in parsed[1:]]
    return "\n".join(out)

FINANCIAL_TOOLS = [
    calculate_cagr,
    calculate_growth,
    calculate_roi,
    compare_returns,
    create_comparison_table,
]
