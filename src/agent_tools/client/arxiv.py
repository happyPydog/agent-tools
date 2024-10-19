"""Arxiv client tools."""

from __future__ import annotations

import re
import ssl
import tempfile
from typing import Iterable, cast

import fitz
from arxiv import Client as _ArxivClient
from arxiv import Result as ArxivResult
from arxiv import Search as ArxivSearch
from more_itertools import flatten, unique_everseen
from pydantic import AnyHttpUrl, BaseModel


class Paper(BaseModel):
    title: str
    text: str
    url: AnyHttpUrl
    references: list[Paper] | None = None

    def flatten(self) -> list[Paper]:
        papers = [self]
        if self.references:
            for paper in self.references:
                papers.extend(paper.flatten())
        return papers


class ArxivClient:

    def __init__(self) -> None:
        self.client = _ArxivClient()
        self.ensure_ssl_verified()

    @staticmethod
    def ensure_ssl_verified() -> None:
        ssl._create_default_https_context = ssl._create_unverified_context

    def search_by_url(self, id_list: list[str]) -> Iterable[ArxivResult]:
        search = ArxivSearch(id_list=id_list)
        yield from self.client.results(search=search)

    def search_by_query(self, queries: Iterable[str]) -> Iterable[ArxivResult]:
        for query in queries:
            search = ArxivSearch(query=query, max_results=1)
            yield from self.client.results(search=search)

    @staticmethod
    def extract_id(url: str) -> str | None:
        match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
        return match.group(1) if match else None

    @staticmethod
    def parse_references(text: str) -> list[str]:
        arxiv_urls = re.findall(r"(https?://arxiv\.org/abs/\d{4}\.\d{4,5}(v\d+)?)", text)
        return [match[0] for match in arxiv_urls]

    def fetch_papers_by_url(self, urls: Iterable[str]) -> list[Paper]:
        id_list = list(filter(None, (self.extract_id(url) for url in urls)))

        papers = []
        for paper in self.search_by_url(id_list):
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = paper.download_pdf(dirpath=temp_dir)
                doc = cast(fitz.Document, fitz.open(pdf_path))
                text = "".join([page.get_text() for page in doc])  # type: ignore

            papers.append(Paper(title=paper.title, text=text, url=paper.entry_id))  # type: ignore

        return papers

    def fetch_papers_with_references_by_url(self, urls: Iterable[str]) -> list[Paper]:
        parent_papers = self.fetch_papers_by_url(urls)

        for paper in parent_papers:
            reference_urls = self.parse_references(paper.text)
            if reference_urls:
                referenced_papers = self.fetch_papers_by_url(reference_urls)
                paper.references = referenced_papers

        return parent_papers

    def fetch_papers_by_query(self, queries: Iterable[str]) -> list[Paper]:
        papers = []
        for requested_query in queries:
            results = self.search_by_query([requested_query])
            for paper in results:
                if paper.title.lower() != requested_query.lower():
                    continue
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_path = paper.download_pdf(dirpath=temp_dir)
                    doc = cast(fitz.Document, fitz.open(pdf_path))
                    text = "".join([page.get_text() for page in doc])  # type: ignore

                papers.append(Paper(title=paper.title, text=text, url=paper.entry_id))  # type: ignore
                break
        return papers

    def fetch_papers_with_references_by_query(self, queries: Iterable[str]) -> list[Paper]:
        parent_papers = self.fetch_papers_by_query(queries)

        for paper in parent_papers:
            reference_urls = self.parse_references(paper.text)
            if reference_urls:
                referenced_papers = self.fetch_papers_by_url(reference_urls)
                paper.references = referenced_papers

        return parent_papers

    def download_papers_by_url(self, urls: str | Iterable[str], save_dir: str) -> None:
        id_list = list(filter(None, (self.extract_id(url) for url in urls)))

        for paper in self.search_by_url(id_list):
            filename = re.sub(r"\W+", "_", paper.title) + ".pdf"
            paper.download_pdf(dirpath=save_dir, filename=filename)

    def download_papers_by_query(self, queries: str | Iterable[str], save_dir: str) -> None:
        for requested_query in queries:
            results = self.search_by_query([requested_query])
            for paper in results:
                if paper.title.lower() != requested_query.lower():
                    continue
                filename = re.sub(r"\W+", "_", paper.title) + ".pdf"
                paper.download_pdf(dirpath=save_dir, filename=filename)


arxiv_client = ArxivClient()


def fetch_papers_by_url(urls: str | Iterable[str], parse_reference: bool = False) -> list[Paper]:
    urls = flatten([urls])

    if parse_reference:
        return arxiv_client.fetch_papers_with_references_by_url(urls)

    return arxiv_client.fetch_papers_by_url(urls)


def fetch_papers_by_query(queries: str | Iterable[str], parse_reference: bool = False) -> list[Paper]:
    queries = flatten([queries])

    if parse_reference:
        return arxiv_client.fetch_papers_with_references_by_query(queries)

    return arxiv_client.fetch_papers_by_query(queries)


def download_papers_by_url(urls: str | Iterable[str], save_dir: str = "./") -> None:
    urls = flatten([urls])
    arxiv_client.download_papers_by_url(urls, save_dir)


def download_papers_by_query(queries: str | Iterable[str], save_dir: str = "./") -> None:
    queries = flatten([queries])
    arxiv_client.download_papers_by_query(queries, save_dir)


def extract_refs(papers: Iterable[Paper]) -> list[Paper]:
    return list(unique_everseen(flatten([paper.flatten() for paper in papers])))
