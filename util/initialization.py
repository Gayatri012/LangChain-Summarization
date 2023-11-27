import os
import sys

from PromptTemplateClasses.SummarizationTemplate import SummarizationTemplate


def set_openai_api(api_key):
    os.environ['OPENAI_API_TOKEN'] = api_key
    if os.getenv('OPENAI_API_TOKEN') is not None:
        print("Successfully set_openai_api()")
    else:
        print("Environment variable is not set. Please try again")
        raise Exception("set_openai_api variable is not set. Please try again")


if __name__ == "__main__":
    # TODO: Provide users with different options in file types that can be uploaded, not just html document
    if len(sys.argv) == 2:
        user_api_key = sys.argv[1]
        # html_doc_location = sys.argv[2]
    else:
        raise Exception("please provide your open ai api_key as an argument to the program")

    set_openai_api(user_api_key)

    # print("Path of the document to summarized: " + html_doc_location)

    test_text = "Our non-marketable equity securities are investments in privately-held companies without readily " \
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

    summarization_obj = SummarizationTemplate()
    test_docs, number_of_tokens = summarization_obj.split_input_text_to_tokens(test_text)

    summarization_obj.run_summary(test_docs, number_of_tokens)
