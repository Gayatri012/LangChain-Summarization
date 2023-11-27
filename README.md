# LangChain-Summarization
Use LangChain to extract information from 10-K/ 10-Q SEC filings and create summaries of the extracted document

1. A test function to check the OpenAI API's summarization performance has been added to the tests/test_openai.py. This script can be run as a main class by providing your OpenAI api_token as an input parameter.
   1. Run the program as below: <br>
   /usr/bin/python3 <full path of the project>/LangChain-Summarization/tests/test_openai.py <open_ai_api_token_key>



Thanks to Johni for his article on Medium, which is beginner friendly and thorough. It helped me immensely in this project. <br>
https://medium.com/@johnidouglasmarangon/how-to-summarize-text-with-openai-and-langchain-e038fc922af