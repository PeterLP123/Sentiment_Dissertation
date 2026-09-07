# Security and sensitive-data reporting

## Supported version

Security and data-exposure fixes are applied to the current `main` branch. Historical
research artifacts are retained for provenance, but a correction may quarantine or
redact an artifact when continued publication would expose credentials or restricted
data.

## Reporting a problem

Do not open a public issue containing a credential, private endpoint, licensed news
text, provider response, personal information, or an exploitable vulnerability. Use
GitHub's private vulnerability-reporting flow from the repository's **Security** tab
when it is available. Otherwise, contact the repository owner privately through the
contact method on the owner's GitHub profile. Include only the minimum information
needed to locate and assess the problem.

Useful reports identify the affected path and revision, explain the exposure or
impact, and give safe reproduction steps. Do not attach a full restricted dataset or
paste a working secret. If a credential may be live, revoke or rotate it before
continuing the investigation.

Questions about permitted reuse or dataset licensing belong under the
[data and rights policy](docs/data_and_rights.md); they are not security reports.
