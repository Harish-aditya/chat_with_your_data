import os

import streamlit as st
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser

from data import load_data

col1, col2 = st.columns(2)
col1.header("Summary")
col2.header("Chart")

class StreamlitCallback(BaseCallback):
    def __init__(self, container) -> None:
        """Initialize callback handler."""
        self.container = container

    def on_code(self, response: str):
        self.container.code(response)


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return


st.write("# Chat with your Data")

os.environ["OPENAI_API_KEY"]="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"


df = load_data(r"./data")

with st.expander("🔎 Data Preview"):
    st.write(df.tail(10))

query = st.text_area("🗣️ Chat with Data")
# container = st.container()


if query:
    llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
    query_engine = SmartDataframe(
        df
        ,
        config={
            "llm": llm,
            "response_parser": StreamlitResponse
            # "callback": StreamlitCallback(container),
        },
    )

    answer = query_engine.chat(query)
    print(answer)