def init_llm_chain():
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-16k",
        openai_api_key="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",
    )
