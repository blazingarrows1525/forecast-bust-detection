"""GenAI layer: guarded, tool-calling assistance over the bulletin store.

Default OFF (D-015). Importing this package must never make a network call and
must never fail on a machine without AWS credentials, so the Bedrock client is
imported lazily inside fbd.genai.client.
"""
from fbd.genai.settings import GenAISettings, load

__all__ = ["GenAISettings", "load"]
