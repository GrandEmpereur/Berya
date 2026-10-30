"""make_llms(backend) : la seule différence de code entre les deux cas d'étude."""
import config as C


def make_llms(backend: str = C.BACKEND):
    if backend == "local":
        from langchain_ollama import ChatOllama
        gen  = ChatOllama(model=C.LOCAL_GEN_MODEL,  temperature=0, num_ctx=C.LOCAL_GEN_NUM_CTX,  reasoning=False)
        util = ChatOllama(model=C.LOCAL_UTIL_MODEL, temperature=0, num_ctx=C.LOCAL_UTIL_NUM_CTX, reasoning=False)
    elif backend == "claude":
        from langchain_anthropic import ChatAnthropic
        gen  = ChatAnthropic(model=C.CLAUDE_GEN_MODEL,  max_tokens=C.GEN_MAX_TOKENS,  thinking=C.CLAUDE_GEN_THINKING)
        util = ChatAnthropic(model=C.CLAUDE_UTIL_MODEL, max_tokens=C.UTIL_MAX_TOKENS, thinking=C.CLAUDE_UTIL_THINKING)
    else:
        raise ValueError(backend)
    return gen, util


def make_judge():
    """Juge des evals : le même pour les trois expériences, quel que soit BERYA_BACKEND."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=C.JUDGE_MODEL, max_tokens=C.JUDGE_MAX_TOKENS, thinking=C.CLAUDE_UTIL_THINKING)


if __name__ == "__main__":
    for backend in ("local", "claude"):
        gen, util = make_llms(backend)
        print(f"{backend:6} gen  = {type(gen).__name__}({gen.model})")
        print(f"{backend:6} util = {type(util).__name__}({util.model})")
    print(f"judge       = {make_judge().model}")
