# Hugging Face Space for Forecast Bust Detection: the published image, unchanged.
#
# The image is built, smoke-tested (including as UID 1000 in read-only mode,
# which is how a Space runs it) and pushed by .github/workflows/publish.yml on
# every push to master. Building from the repo instead would miss the bulletin
# store, which is a Release asset and not in git (DATA.md), and every panel
# would be empty.
#
# After a new image is published, use "Factory rebuild" in the Space settings
# so the Space pulls the new :latest.
FROM ghcr.io/blazingarrows1525/forecast-bust-detection:latest

# A public demo takes no anonymous writes to the forecaster override log.
ENV FBD_READ_ONLY=1
