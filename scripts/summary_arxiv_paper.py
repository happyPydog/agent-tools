"""Summary arXiv paper."""

import io
import os
import tempfile
from typing import Any
from langfuse.openai import OpenAI
from dotenv import load_dotenv
from rich import print
from pydantic import BaseModel, Field, TypeAdapter
from tenacity import retry, stop_after_attempt
from agent_tools.prompt_func import openai_prompt
from agent_tools.client.arxiv import Paper, fetch_papers_by_url
from agent_tools.utilities import extract_json_content

load_dotenv

urls = [
    "https://arxiv.org/abs/2410.18057",
]

llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


class Section(BaseModel):
    section: str = Field(
        ...,
        description="The title or heading of the section, representing the main topic discussed in this part of the paper.",
    )
    content: str = Field(
        ...,
        description="The entire text content of the section, providing all details as presented in the paper without reduction or summarization.",
    )
    ref_fig: list[str] = Field(
        ...,
        description="A list of figure references within this section, indicating any figures mentioned. Use the format Figure-<figure_number>-<page>. Format examples: ['Figure-1-1', 'Figure-2-2'].",
    )
    ref_tb: list[str] = Field(
        ...,
        description="A list of table references within this section, indicating any tables mentioned. Use the format Figure-<figure_number>-<page>. Format examples: ['Table-1-1', 'Table-2-2'].",
    )


# class Section(BaseModel):
#     section: str
#     content: str
#     page: list[str]


class PaperSummarizer:

    def __init__(self):
        self.pdf_output_dir = tempfile.mkdtemp()

    def batch_summary(self, urls: list[str]): ...
    def summary(self, url: str): ...

    def __call__(self, urls: list[str] | str):
        if isinstance(urls, str):
            urls = [urls]

        papers = fetch_papers_by_url(urls)
        for paper in papers:
            # 1. Extract sections from the paper
            sections = self.parse_sections(paper)
            print(f"Extracted sections: {sections}")

            # 2. Summary each sections
            # TODO: consider parse the ref_fig and ref_tb

            # 3. Find insights of each sections and quotes

            # 4. Organize the sections and insights into an overall summary

    @retry(stop=stop_after_attempt(3))
    def parse_sections(self, paper: Paper) -> list[Section]:
        section_obj = self.extract_sections(paper.text)
        section_obj = extract_json_content(section_obj)
        return TypeAdapter(list[Section]).validate_python(section_obj)

    @openai_prompt(
        ("system", "You are a helpful AI assistant."),
        (
            "user",
            "## Task:\n"
            "Given the content of a paper in triple backticks, organize each section into a JSON object format where each element contains:\n"
            "- `section`: The name of the section, capturing the main topic or heading of the section.\n"
            "- `content`: The full content of the section, presented exactly as in the paper without reduction or summarization.\n"
            "- `ref_fig`: A list of figure references in this section. Each reference must follow the format 'figure-<page>-<number>' (e.g., 'figure-20-7' for the seventh figure on page 20).\n"
            "- `ref_tb`: A list of table references in this section. Each reference must follow the format 'table-<page>-<number>' (e.g., 'table-15-3' for the third table on page 15).\n\n"
            "Ensure that each section is represented as a structured JSON object, without reducing or summarizing the content. "
            "Do not include the references section.\n",
        ),
        (
            "user",
            '### Response Format:\n```json\n[{{"section": "str", "content": "str", "ref_fig": ["str"], "ref_tb": ["str"]}}]\n```',
        ),
        ("user", "Paper content: ```{text}```"),
        model_name="gpt-3.5-turbo",
    )
    def extract_sections(self, text: str) -> str: ...

    def partition(self, file: io.BytesIO) -> list[Element]:
        return partition_pdf(
            file=file,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=self.output_dir,
        )


def main():
    paper_summarizer = PaperSummarizer()
    section_list = paper_summarizer(urls)
    print(section_list)


if __name__ == "__main__":
    main()
