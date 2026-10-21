"""Expose three educational cybersecurity tools through MCP."""

from fastmcp import FastMCP

mcp = FastMCP("Homework 4 Cybersecurity Tools")

COMMON_PORTS = {
    21: "FTP",
    22: "SSH",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    110: "POP3",
    143: "IMAP",
    443: "HTTPS",
    3389: "RDP",
}

SECURITY_TERMS = {
    "phishing": "Deceptive messages that trick people into revealing "
                "information or taking harmful actions.",
    "malware": "Software designed to damage systems, steal data, "
               "or perform unauthorized actions.",
    "ransomware": "Malware that blocks access to data or systems "
                  "and demands payment.",
    "firewall": "A security control that filters network traffic "
                "according to configured rules.",
    "encryption": "Transforming readable data into ciphertext "
                  "that requires a key to decrypt.",
    "vulnerability": "A weakness that could be exploited to cause harm.",
    "mfa": "Multi-factor authentication: verifying identity using "
           "two or more distinct types of evidence.",
}


@mcp.tool()
def check_port(port: int) -> dict:
    """Look up a common port association without scanning a network.

    Args:
        port: An integer from 1 through 65535.

    Returns:
        The port, whether it appears in this tool's small reference
        table, and its commonly associated service.
    """
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")

    service = COMMON_PORTS.get(port)
    return {
        "port": port,
        "recognized": service is not None,
        "service": service,
        "note": "A port association does not verify a running service.",
    }


@mcp.tool()
def explain_security_term(term: str) -> dict:
    """Explain a cybersecurity term from a small built-in glossary.

    Args:
        term: A term such as phishing, encryption, or MFA.

    Returns:
        The normalized term and its definition, or available terms
        when no definition is found.
    """
    normalized = term.strip().casefold()
    if not normalized:
        raise ValueError("Enter a nonempty cybersecurity term.")

    definition = SECURITY_TERMS.get(normalized)
    if definition is None:
        return {
            "term": normalized,
            "definition": "Term not found in the built-in glossary.",
            "available_terms": sorted(SECURITY_TERMS),
        }
    return {"term": normalized, "definition": definition}


@mcp.tool()
def calculate_risk_score(likelihood: int, impact: int) -> dict:
    """Calculate a simplified educational risk score.

    Args:
        likelihood: Likelihood rating from 1 (lowest) to 5 (highest).
        impact: Impact rating from 1 (lowest) to 5 (highest).

    Returns:
        The product of the ratings and a category: low (1–4),
        medium (5–9), high (10–16), or critical (17–25).
        This illustrative scale is not CVSS.
    """
    if not 1 <= likelihood <= 5 or not 1 <= impact <= 5:
        raise ValueError("Likelihood and impact must each be 1 through 5.")

    score = likelihood * impact
    if score <= 4:
        category = "low"
    elif score <= 9:
        category = "medium"
    elif score <= 16:
        category = "high"
    else:
        category = "critical"

    return {
        "likelihood": likelihood,
        "impact": impact,
        "score": score,
        "category": category,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
