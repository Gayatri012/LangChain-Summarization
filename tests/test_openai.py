import sys
from openai import OpenAI


# Reference function without LangChain and with only Open AI to see how the API responds to the summarization request
def test_openai_summarize_text(prompt_text, api_key):
    print("Inside test_openai_summarize_text()")
    # This code is for v1 of the openai package: pypi.org/project/openai
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant who will assist in "
                                                "summarizing the input text into 5 important points"},
                  {"role": "user", "content": prompt_text}],
        temperature=0.1,
        max_tokens=1024
    )
    return response


if __name__ == "__main__":
    # In order for this to work, please provide your open ai api key as an input to the program
    if len(sys.argv) == 2:
        user_api_key = sys.argv[1]
    else:
        raise Exception("please provide your open ai api_key as an argument to the program")

    # Reference: The input_text is taken from the publicly released 10-K filings by Meta in Feb 2023.
    input_text = "Our non-marketable equity securities are investments in privately-held companies without readily " \
                 "determinable fair values. We elected to account for most of our non-marketable equity securities " \
                 "using the measurement alternative, which is cost, less any impairment, adjusted for changes in " \
                 "fair value resulting from observable transactions for identical or similar investments of the same " \
                 "issuer. We perform a qualitative assessment at each reporting date to determine whether there are " \
                 "triggering events for impairment. The qualitative assessment considers factors such as, but not " \
                 "limited to, the investee's financial condition and business outlook; industry and sector " \
                 "performance; economic or technological environment; and other relevant events and factors affecting" \
                 " the investee. Valuations of our non-marketable equity securities are complex due to the lack of " \
                 "readily available market data and observable transactions. Uncertainties in the global economic " \
                 "climate and financial markets could adversely impact the valuation of these companies we invest in " \
                 "and, therefore, result in a material impairment or downward adjustment in our investments. Our " \
                 "total non-marketable equity securities had a carrying value of $6.20 billion and $6.78 billion as " \
                 "of December 31, 2022 and 2021, respectively."
    print("Input text: \n ", input_text)

    summary_text = test_openai_summarize_text(input_text, user_api_key)
    print("Summary text: \n", summary_text.choices[0].message.content)
