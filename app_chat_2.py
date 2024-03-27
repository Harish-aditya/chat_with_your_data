import os

import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from data import load_data




class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        col2.dataframe(result["value"])
        return

    def format_plot(self, result):
        col2.image(result["value"],use_column_width=True)
        return

    def format_other(self, result):
        col2.write(result["value"])
        return
    
    def format_answer(self, result):
        col2.write(result["value"])
        return


st.write("# Chat with your Data")

# key = st.text_input("Enter your Key")

os.environ["OPENAI_API_KEY"]=st.secrets['OPENAI_API_KEY']


df = load_data(r"./data")


with st.expander("🔎 Data Preview"):
    st.write(df.tail(10))

query = st.text_area("🗣️ Chat with Data")
container = st.container()

col1, col2 = st.columns(2)
col1.header("Summary")
col2.header("Chart")

if query:
    llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"],llm="gpt-4-0613",temperature=0.5)
    query_engine = SmartDataframe(
        df        ,
        config={
            "llm": llm,
            "response_parser": StreamlitResponse,
        },
    )

    template="""
    you need to analysis the data and use pandas to get the output.
    
    {context}
    
    Question: {question}
    Answer: 

    """

    context = """
    You are the data analysit. you need to analysis and give a paragraph or points reply to the manager .you should be able to explain the data clearly.
    Strictly avoid charts.

    Information about Data
    Row ID - Unique row Identifier
    Order ID - Unqiue Order Identifier
    Order Date - Date of the order
    Ship Date - Date of the shipping of the placed order
    Customer - Name of the Customer who places order.
    Segment - Group of type of Customer
    Country/Region - Country or Region of the Customer
    City - City of the Customer
    State/Province - State or Province of the Customer
    Postal Code - Postal Code of the Customer
    Region - Region of the Country the Customer resides in.
    Product ID - Unqiue Identifier of the Product in the Store
    category - Category of the Product in the store
    Sub Category - Sub Category of the product in the store
    Product Name - Name of the Product in the store
    Sales - Sales indicates sum amount store gets off the order
    Quantity - Units of the product in the order
    Discount - Decimal form of percentange of Discount given by store for the customer.
    Profit - Profit made by the store from that particular order.
    """

    context_pandas = """
    You are the data analysit. you need to analysis and give a paragraph or points reply to the manager .you should be able to explain the data clearly.
    You should always generate charts to explain. Mention data labels to the chart. Always use seaborn to generate Chart.
    """

    pandas_txt = query_engine.chat(context_pandas + ' ' + query)

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    agent= create_pandas_dataframe_agent(ChatOpenAI(temperature=0.7, model="gpt-4-0613"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    kwargs={"prompt":PROMPT}
    )

    

    col1.write(agent.invoke(query,config={"response_parser": StreamlitResponse})['output'])



    