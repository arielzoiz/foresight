"""foresight -- a caller-agnostic prompt-enhancement proxy.

To its caller, foresight *is* a language model: it speaks the OpenAI
chat-completions protocol, so a harness configured with
``base_url = http://host:8000/v1`` cannot distinguish it from vLLM or a hosted
API. Inside that single request/response it consults an auxiliary model about
plausible future work, rewrites the task prompt to carry that context, and
forwards it to the target model.

It is *not* an agent: no tools, no loop, it never edits files.

The control arm of the experiment is the same pipeline with a ``passthrough``
builder, so both arms traverse identical code and differ only in the prompt.
"""

__version__ = "0.1.0"
