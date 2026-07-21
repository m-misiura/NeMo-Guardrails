# schemas/

**openai-spec.yml** — Vendored OpenAI API spec (v2.3.0, OpenAPI 3.1.0).
Source: https://github.com/openai/openai-openapi

To update:

```bash
curl -fsSL https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml -o schemas/openai-spec.yml
python scripts/openai_coverage.py --update
```

**openai-coverage.json** — Auto-generated coverage baseline. Do not edit manually.
