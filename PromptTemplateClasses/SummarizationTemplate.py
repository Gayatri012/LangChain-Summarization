import tiktoken
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import textwrap
from time import monotonic


class SummarizationTemplate:
    def __init__(self):
        # TODO: Can try to summarize using other models in OpenAI as well as using other LLMs
        print(os.getenv('OPENAI_API_TOKEN'))
        self.summarization_model_name = "gpt-3.5-turbo"
        # Read this value from the open ai doc
        self.gpt_35_turbo_max_tokens = 4097

        self.summarization_model = ChatOpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_API_TOKEN'),
                                         model_name=self.summarization_model_name)

        prompt_template = """Write a concise summary of the input text into 5 important points:

                        {text}

                        CONCISE SUMMARY : """

        self.prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    def summarize_text(self, text):
        print("Inside summarize_text()")
        llm_chain = LLMChain(
            prompt=f"Please summarize the following text:\n{text}\n\nSummary:",
            llm=self.summarization_model
        )

        print(llm_chain.run(text))

    def split_input_text_to_tokens(self, input_text):
        model_name = self.summarization_model_name
        # Splitting the input text by recursively dividing the text at specific characters
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(model_name=model_name)

        texts = text_splitter.split_text(input_text)

        # Converting the raw text into chunks of Documents
        docs = [Document(page_content=t) for t in texts]
        print("len(docs): ", len(docs))

        # Finding the number of tokens in the input text to know how many times to split the input_text

        # tiktoken is a fast BPE (Byte-pair encoding) tokenizer for use with OpenAI models. It is reversible & lossless
        encoding = tiktoken.encoding_for_model(self.summarization_model_name)
        num_tokens = len(encoding.encode(input_text))
        print("len(num_tokens): ", num_tokens)

        return docs, num_tokens

    def run_summary(self, input_docs, num_tokens):
        print("Inside run_summary()")
        verbose = True

        if num_tokens < self.gpt_35_turbo_max_tokens:
            chain = load_summarize_chain(self.summarization_model, chain_type="stuff", prompt=self.prompt,
                                         verbose=verbose)
        else:
            chain = load_summarize_chain(self.summarization_model, chain_type="map_reduce", map_prompt=self.prompt,
                                         combine_prompt=self.prompt,
                                         verbose=verbose)

        start_time = monotonic()
        summary = chain.run(input_docs)

        print(f"Chain type: {chain.__class__.__name__}")
        print(f"Run time: {monotonic() - start_time}")
        print(f"Summary: {textwrap.fill(summary, width=100)}")

