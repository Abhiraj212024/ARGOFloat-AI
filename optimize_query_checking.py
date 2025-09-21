from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3:8b-instruct-q4_K_M")

def extract_filters(user_query: str) -> str:
    system_prompt = """
    You are an assistant that extracts numerical filters from oceanographic queries.
    Return a JSON dictionary of the form:
    {{
        "column_name": [min_value, max_value],
        ...
    }}
    Rules:
    - Use numbers for bounds if specified.
    - Use null for any bound that is not specified.
    - Only include columns mentioned in the query.
    - Only return a JSON string and nothing else.
    Example:
    Input: "Show me profiles with temperature above 5 and salinity below 35."
    Output: {{"temperature": [5, null], "salinity": [null, 35]}}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    filters_str = chain.invoke({"question" : user_query})
    
    return filters_str

    # try:
    #     import json
    #     filters_dict = json.loads(filters_str)
    #     return filters_dict
    # except json.JSONDecodeError:
    #     return {"error": "Failed to parse JSON"}

print(extract_filters("Show me all profiles with temperature between 5 and 15 degrees and salinity above 30."))