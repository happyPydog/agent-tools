import asyncio
import os
import time

from dotenv import load_dotenv
from magentic import chatprompt
from magentic.chat_model.message import SystemMessage, UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from pydantic import BaseModel, Field

load_dotenv()

model = OpenaiChatModel(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_ENDPOINT"),
    model="gpt-4o",
)


class TraditionalChineseTranslationRequest(BaseModel):
    text: str = Field(..., title="Text to translate")
    translation: str = Field(..., title="Translated text")


@chatprompt(
    SystemMessage("You are a helpful assistant."),
    UserMessage("Translate the given text to TRADITIONAL CHINESE."),
    UserMessage("Text: {text}"),
    model=model,
)
async def translate_text_to_traditional_chinese(text: str) -> TraditionalChineseTranslationRequest: ...


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
    tasks = [translate_text_to_traditional_chinese(title) for title in titles]
    translated_texts = await asyncio.gather(*tasks)
    print(f"Time taken: {time.time() - start}")
    for translated in translated_texts:
        print(translated)


if __name__ == "__main__":
    asyncio.run(main())
