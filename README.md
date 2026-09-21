# Jevstiller

Jev still too expensive and slow for you? Distill it on the fly.

Jevstiller sits between a repeated classification task and Jev. At first it forwards everything to Jev; from Jev's own answers it trains a small task-specific model, verifies that the small model agrees with Jev within a user-set budget, and then answers most requests itself — sending Jev only the uncertain ones plus a small audit sample.

See [DESIGN.md](DESIGN.md) for the full design.
