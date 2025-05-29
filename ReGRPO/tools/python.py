def python(code: str) -> str:
    """Executes a block of Python code and returns any text printed to stdout.
    Only the following libraries may be imported inside the snippet:
    astropy, biopython, networkx, numpy, scipy, and sympy.
    
    Args:
        code: A block of valid Python source code.
    
    Returns:
        The captured output (trimmed to 1000 characters) or an error message.
    
    Examples:
        {"code": "import numpy as np; print(np.arange(3) * 2)"} -> "[0 2 4]"
        {"code": "import sympy as sp; x = sp.symbols('x'); print(sp.integrate(sp.sin(x), x))"} -> "-cos(x)"
        {"code": "import networkx as nx; g = nx.path_graph(4); print(g.number_of_edges())"} -> "3"
    """
    import subprocess

    try:
        # Run the snippet in a separate process with a 10-second timeout
        completed = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Surface any stderr as an error message
        if completed.stderr:
            return f"Error: {completed.stderr.strip()}"

        output = completed.stdout or ""

        if len(output) > 1000:
            output = output[:1000] + "... (truncated to 1000 chars)"

        return output.strip()

    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 10 seconds"
