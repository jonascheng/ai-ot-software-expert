# a python code to recognize a software which is designed for
# operational technology (OT) and industrial control systems (ICS)
# environments. The list of software are input from a csv file
import csv
import os
import sys

import openai
from langchain.llms import AzureOpenAI
from langchain import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema import OutputParserException

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")


# open a csv file and read the list of software
def read_csv_file(csv_file):
    with open(csv_file, 'r') as f:
        # reader = csv.reader(f)
        # software_list = list(reader)
        software_list = csv.DictReader(f)
        # remove csv header
        # software_list.pop(0)
        # return software_list
        for row in software_list:
            yield row


# check if the software is designed for OT/ICS environments.
# this function will invoke openai to do the job
def check_software(caption, vendor):
    # check if the software is designed for OT/ICS environments
    # return a scale 0 to 5, higher scale means highly confident to be a OT/ICS software.
    # return additional description of the capability of this software in 1 line.

    deployment_name = "text-davinci-003"
    # deployment_name = "gpt-35-turbo"
    llm = AzureOpenAI(deployment_name=deployment_name,
                      temperature=0,
                      model_kwargs={
                          "api_type": "azure",
                          "api_version": "2022-12-01",
                          "api_base": os.getenv('OPENAI_API_BASE'),
                          "api_key": os.getenv("OPENAI_API_KEY"),
                      })

    # define the output parser
    response_schemas = [
        ResponseSchema(
            name="scale",
            description=
            "a confidence scale 0 to 5, higher scale means highly confident to be a OT/ICS software"
        ),
        ResponseSchema(
            name="brief",
            description=
            "a brief description of the software capabilityin 1 line"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(
        response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # define the prompt template
    template = """
    You're an expert of operational technology (OT) and industrial control system (ICS), and fully comprehend those software that will be installed on HMI and SCADA.
    Please tell me if the software {caption} (vendor {vendor}) is designed for OT/ICS along with a confidence scale 0 to 5, higher scale means highly confident to be a OT/ICS software.
    In addition to the scale, please also provide a brief description of the software capabilityin 1 line. Please respond in the following format: {format_instructions}
    """
    prompt = PromptTemplate(
        input_variables=["caption", "vendor", "format_instructions"],
        template=template,
    )
    final_prompt = prompt.format(caption=caption,
                                 vendor=vendor,
                                 format_instructions=format_instructions)

    # invoke openai to do the job and catch exception
    try:
        print("-----------")
        print(f"Final Prompt: {final_prompt}")
        resp = llm(final_prompt)
        # replace '\' with '$' to avoid ParserException: Got invalid JSON object. Error: Invalid \escape
        resp = resp.replace('\\', '$')
        print(f"LLM Output: {resp}")
        ret = output_parser.parse(resp)
        print(f"Output Parsed: {ret}")
        print("-----------\n")
    except OutputParserException as e:
        print(f"Error: {e}")
        ret = {"scale": -1, "brief": "OutputParserException"}
    
    return ret["scale"], ret["brief"]


# write the result to a csv file
f = open('output.csv', 'w')
header = ['scale', 'caption', 'vendor', 'brief']
writer = csv.DictWriter(f, fieldnames=header)
writer.writeheader()

# open a csv file and read the list of software
software_list = read_csv_file('software.csv')

# loop through the list of software and check if the software is designed for
# OT/ICS environments.
for software in software_list:
    # clone the software
    result = software

    # remove empty column and 'name' from the dict
    result.pop('')
    result.pop('name')

    scale, brief = check_software(software['caption'], software['vendor'])

    # add the result to the dict
    result['scale'] = scale
    result['brief'] = brief

    # write the result to a csv file
    writer.writerow(result)
    # flush the buffer
    f.flush()

# close the csv file
f.close()