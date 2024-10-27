import asyncio
import os
import time

from dotenv import load_dotenv
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain_openai.chat_models import ChatOpenAI
from langfuse.callback import CallbackHandler as LangFuseCallbackHandler
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_ENDPOINT"),
    model="gpt-4o",
)
langfuse_callback_handler = LangFuseCallbackHandler(tags=["async_translate_langchain"])


class TraditionalChineseTranslationRequest(BaseModel):
    text: str = Field(..., title="Text to translate")
    translation: str = Field(..., title="Translated text")


async def translate_text_to_traditional_chinese(
    texts: str | list[str],
    llm: ChatOpenAI,
    langfuse_handler: LangFuseCallbackHandler,
) -> list[TraditionalChineseTranslationRequest]:
    if isinstance(texts, str):
        texts = [texts]
    inputs = [{"text": text} for text in texts]

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
            HumanMessagePromptTemplate.from_template("Translate the given text to Traditional Chinese.\nText: {text}"),
        ]
    )

    # Use structured output
    llm_structured = llm.with_structured_output(TraditionalChineseTranslationRequest)
    chain = prompt | llm_structured

    # Collect the results using `abatch`
    results = await chain.abatch(inputs=inputs, config={"callbacks": [langfuse_handler]})

    print(f"Results: {results}")
    return results


titles = [
    "Impeding LLM-assisted Cheating in Introductory Programming Assignments via Adversarial Perturbation",
    "Tokenization Is More Than Compression",
    "GoldCoin: Grounding Large Language Models in Privacy Laws via Contextual Integrity Theory",
    "C3PA: An Open Dataset of Expert-Annotated and Regulation-Aware Privacy Policies to Enable Scalable Regulatory Compliance Audits",
    "Where is the signal in tokenization space?",
    "Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models",
    "Finding Blind Spots in Evaluator LLMs with Interpretable Checklists",
    "Delving into Qualitative Implications of Synthetic Data for Hate Speech Detection",
    '"They are uncultured": Unveiling Covert Harms and Social Threats in LLM Generated Conversations',
    "BiasWipe: Mitigating Unintended Bias in Text Classifiers through Model Interpretability",
    "De-Identification of Sensitive Personal Data in Datasets Derived from IIT-CDIP",
    "A Reflective LLM-based Agent to Guide Zero-shot Cryptocurrency Trading",
]


async def main():
    start = time.time()
    translated_texts = await translate_text_to_traditional_chinese(
        texts=titles,
        llm=model,
        langfuse_handler=langfuse_callback_handler,
    )
    print(f"Time taken: {time.time() - start}")
    for translated in translated_texts:
        print(f"Original: {translated.text}\nTranslated: {translated.translation}\n")


if __name__ == "__main__":
    asyncio.run(main())
